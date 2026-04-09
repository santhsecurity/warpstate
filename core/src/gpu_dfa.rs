//! GPU DFA matcher using wgpu compute shaders for regex pattern matching.
//!
//! **Deprecated**: Use [`PersistentMatcher`](crate::persistent::PersistentMatcher),
//! [`SmemDfaMatcher`](crate::gpu_smem::SmemDfaMatcher), or
//! [`AlgebraicDfaMatcher`](crate::algebraic::AlgebraicDfaMatcher) instead.
//! Those backends are faster and correctly integrated with the auto-router.
#[allow(deprecated)]
use crate::config::AutoMatcherConfig;
use crate::dfa::RegexDFA;
use crate::gpu::adapter_is_unsupported;
use crate::pattern::PatternSet;
use crate::shader;
use crate::{
    error::{Error, Result},
    Match, Matcher,
};
use wgpu::util::DeviceExt;

/// Maximum number of matches to record.
const MAX_MATCHES: u32 = 1_048_576;

/// GPU-accelerated Dense DFA Regex matcher.
///
/// **Deprecated**: Use [`PersistentMatcher`](crate::persistent::PersistentMatcher),
/// [`SmemDfaMatcher`](crate::gpu_smem::SmemDfaMatcher), or
/// [`AlgebraicDfaMatcher`](crate::algebraic::AlgebraicDfaMatcher) instead.
#[deprecated(note = "Use PersistentMatcher, SmemDfaMatcher, or AlgebraicDfaMatcher instead")]
pub struct GpuDfaMatcher {
    dfa: RegexDFA,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    transition_table_buf: wgpu::Buffer,
    match_list_pointers_buf: wgpu::Buffer,
    match_lists_buf: wgpu::Buffer,
    pattern_lengths_buf: wgpu::Buffer,
    start_state: u32,
    class_count: u32,
    eoi_class: u32,
    byte_classes: [u32; 256],
    max_input_size: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    input_len: u32,
    start_state: u32,
    max_matches: u32,
    class_count: u32,
    eoi_class: u32,
    _padding0: [u32; 3],
    // WGSL uses vec4<u32> groups for alignment — represent as 64 x [u32;4]
    byte_classes: [[u32; 4]; 64],
}

impl GpuDfaMatcher {
    /// Compile regex patterns into a GPU compute pipeline.
    pub async fn new(patterns: &PatternSet, config: AutoMatcherConfig) -> Result<Self> {
        let regex_dfa = patterns.compiled_regex_dfa()?;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or(Error::NoGpuAdapter)?;
        if adapter_is_unsupported(&adapter.get_info()) {
            return Err(Error::NoGpuAdapter);
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| Error::GpuDeviceError {
                reason: e.to_string(),
            })?;

        // Format transition table byte array
        let transition_bytes = bytemuck::cast_slice(&regex_dfa.transition_table);

        let transition_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dfa transition table"),
            contents: transition_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader_source = shader::generate_regex_dfa_shader();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dfa compute shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dfa bind group layout"),
            entries: &[
                storage_entry(0, true),  // input
                storage_entry(1, true),  // transition table
                storage_entry(2, true),  // match_list_pointers
                storage_entry(3, true),  // match_lists
                storage_entry(4, false), // match output
                storage_entry(5, false), // match count (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(7, true), // pattern lengths
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dfa pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dfa pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let match_list_pointers_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dfa match list pointers"),
                contents: bytemuck::cast_slice(&regex_dfa.match_list_pointers),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let match_lists_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dfa match lists"),
            contents: bytemuck::cast_slice(&regex_dfa.match_lists),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pattern_lengths_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dfa pattern lengths"),
            contents: bytemuck::cast_slice(&regex_dfa.pattern_lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Ok(Self {
            dfa: regex_dfa.clone().into_owned(),
            device,
            queue,
            pipeline,
            bind_group_layout,
            transition_table_buf,
            match_list_pointers_buf,
            match_lists_buf,
            pattern_lengths_buf,
            start_state: regex_dfa.start_state,
            class_count: regex_dfa.class_count,
            eoi_class: regex_dfa.eoi_class,
            byte_classes: regex_dfa.byte_classes,
            max_input_size: config.configured_gpu_max_regex_input_size(),
        })
    }
}

impl Matcher for GpuDfaMatcher {
    #[allow(clippy::too_many_lines)]
    async fn scan(&self, data: &[u8]) -> matchkit::Result<Vec<Match>> {
        let mut matches = vec![Match::from_parts(0, 0, 0); 1024]; // Standard DFA match capacity
        let n = self.dfa.scan_native(data, &mut matches).map_err(|e| {
            matchkit::Error::PatternCompilationFailed {
                reason: e.to_string(),
            }
        })?;
        matches.truncate(n);
        Ok(matches)
    }
}

impl GpuDfaMatcher {
    async fn read_matches(
        &self,
        count_staging: &wgpu::Buffer,
        match_staging: &wgpu::Buffer,
    ) -> Result<Vec<Match>> {
        let required_match_bytes = u64::from(MAX_MATCHES) * 16;
        if match_staging.size() < required_match_bytes {
            return Err(Error::GpuDeviceError {
                reason: format!(
                    "GPU returned inconsistent match count staging buffer size {} for expected {} bytes",
                    match_staging.size(),
                    required_match_bytes
                ),
            });
        }

        // Map and read
        let (tx, rx) = std::sync::mpsc::channel();
        count_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            if let Ok(result) = rx.try_recv() {
                if result.is_err() {
                    return Err(Error::BufferMapFailed);
                }
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::GpuDeviceError {
                    reason: "GPU DFA count readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let count_data = count_staging.slice(..).get_mapped_range();
        let counts: &[u32] = bytemuck::cast_slice(&count_data);
        if counts.len() < 2 {
            drop(count_data);
            count_staging.unmap();
            return Err(Error::GpuDeviceError {
                reason: "GPU DFA count buffer too small".to_string(),
            });
        }
        let match_count = counts[0] as usize;
        let overflow = counts[1];
        drop(count_data);
        count_staging.unmap();

        if overflow != 0 {
            return Err(Error::MatchBufferOverflow {
                count: match_count,
                max: MAX_MATCHES as usize,
            });
        }
        if match_count == 0 {
            return Ok(Vec::new());
        }

        let (tx, rx) = std::sync::mpsc::channel();
        match_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            if let Ok(result) = rx.try_recv() {
                if result.is_err() {
                    return Err(Error::BufferMapFailed);
                }
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::GpuDeviceError {
                    reason: "GPU DFA match readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let match_data = match_staging.slice(..).get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&match_data);

        // Validate match_count against raw buffer length to prevent OOB panics
        // if the GPU returns inconsistent or corrupted results.
        if match_count.saturating_mul(4) > raw.len() {
            return Err(Error::GpuDeviceError {
                reason: format!(
                    "GPU returned inconsistent match count {} for buffer of length {}",
                    match_count,
                    raw.len()
                ),
            });
        }

        let mut matches = Vec::with_capacity(match_count);
        for i in 0..match_count {
            let base = i * 4;
            matches.push(Match {
                pattern_id: raw[base],
                start: raw[base + 1],
                end: raw[base + 2],
                padding: 0,
            });
        }

        drop(match_data);
        match_staging.unmap();

        matches.sort_unstable();
        Ok(matches)
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use crate::AutoMatcherConfig;
    use crate::PatternSet;
    use std::sync::Arc;

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn gpu_dfa_inconsistent_match_count() {
        let ps = PatternSet::builder().regex("a.*b").build().unwrap();
        let config = AutoMatcherConfig::new();
        if let Ok(gpu) = block_on(GpuDfaMatcher::new(&ps, config)) {
            let device = &gpu.device;
            let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue
                .write_buffer(&count_staging, 0, bytemuck::cast_slice(&[100u32, 0u32]));
            let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let result = block_on(gpu.read_matches(&count_staging, &match_staging));
            match result {
                Err(Error::GpuDeviceError { reason }) => {
                    assert!(reason.contains("inconsistent match count"));
                }
                _ => panic!(
                    "Expected GpuDeviceError due to inconsistency, got {:?}",
                    result
                ),
            }
        }
    }
}
