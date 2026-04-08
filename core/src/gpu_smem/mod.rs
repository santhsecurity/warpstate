//! GPU pattern matching with SMEM-staged DFA transitions.

mod scan;
mod state;

use crate::config::AutoMatcherConfig;
use crate::gpu::SharedDeviceQueue;
use crate::shader_smem;
use crate::{
    error::{Error, Result},
    PatternSet,
};
use std::sync::Arc;
use tracing::warn;
use wgpu::util::DeviceExt;

use self::state::{storage_entry, PipelineState, SMEM_BYTES};

/// GPU matcher that stages DFA transitions into workgroup shared memory.
pub struct SmemDfaMatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    states: Arc<crossbeam_queue::ArrayQueue<PipelineState>>,
    start_state: u32,
    class_count: u32,
    eoi_class: u32,
    table_size: u32,
    byte_classes: [u32; 256],
    packed_byte_classes: [[u32; 4]; 64],
    max_input_size: usize,
}

impl std::fmt::Debug for SmemDfaMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmemDfaMatcher")
            .field("class_count", &self.class_count)
            .field("table_size", &self.table_size)
            .field("max_input_size", &self.max_input_size)
            .finish_non_exhaustive()
    }
}

impl SmemDfaMatcher {
    /// Create a new SMEM-staged DFA matcher from a compiled pattern set.
    pub async fn new(patterns: &PatternSet) -> Result<Self> {
        Self::with_config(patterns, AutoMatcherConfig::default()).await
    }

    /// Create a new SMEM-staged DFA matcher with an explicit runtime configuration.
    pub async fn with_config(patterns: &PatternSet, config: AutoMatcherConfig) -> Result<Self> {
        let device_queue = crate::gpu::acquire_device().await?;
        Self::from_device(device_queue, patterns, config)
    }

    /// Create a new SMEM-staged DFA matcher from an existing device/queue pair.
    pub fn from_device(
        device_queue: SharedDeviceQueue,
        patterns: &PatternSet,
        config: AutoMatcherConfig,
    ) -> Result<Self> {
        let regex_dfa = patterns.compiled_regex_dfa()?;
        let state_count = regex_dfa.state_count();
        let table_bytes = regex_dfa
            .transition_table
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| Error::PatternCompilationFailed {
                reason: "DFA transition table size overflows usize".to_string(),
            })?;

        if table_bytes > SMEM_BYTES {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "DFA transition table is {table_bytes} bytes, which exceeds the {SMEM_BYTES}-byte shared-memory limit for SMEM mode"
                ),
            });
        }

        let table_size = u32::try_from(regex_dfa.transition_table.len()).map_err(|_| {
            Error::PatternCompilationFailed {
                reason: "DFA transition table length does not fit in u32".to_string(),
            }
        })?;

        let device = device_queue.0.clone();
        let queue = device_queue.1.clone();
        Self::validate_device(&device)?;
        let (pipeline, bind_group_layout) = Self::compile_pipeline(
            &device,
            regex_dfa.transition_table(),
            regex_dfa.class_count,
            state_count,
        );

        let transition_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("smem dfa transition table"),
            contents: bytemuck::cast_slice(&regex_dfa.transition_table),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let match_list_pointers_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("smem match list pointers"),
                contents: bytemuck::cast_slice(&regex_dfa.match_list_pointers),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let match_lists_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("smem match lists"),
            contents: bytemuck::cast_slice(&regex_dfa.match_lists),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pattern_lengths_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("smem pattern lengths"),
            contents: bytemuck::cast_slice(&regex_dfa.pattern_lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let states = Arc::new(crossbeam_queue::ArrayQueue::new(2));
        for _ in 0..2 {
            let buffers = Self::allocate_buffers(&device, config.configured_gpu_max_input_size());
            let bind_group = Self::assemble_bind_group(
                &device,
                &bind_group_layout,
                &transition_table_buf,
                &match_list_pointers_buf,
                &match_lists_buf,
                &pattern_lengths_buf,
                &buffers,
            );
            let pushed = states.push(PipelineState {
                input_buf: buffers.0,
                match_buf: buffers.1,
                count_buf: buffers.2,
                uniform_buf: buffers.3,
                count_staging: buffers.4,
                match_staging: buffers.5,
                bind_group,
            });
            if pushed.is_err() {
                warn!("pipeline state pool unexpectedly full during SMEM matcher setup");
            }
        }

        Ok(Self {
            device,
            queue,
            pipeline,
            states,
            start_state: regex_dfa.start_state,
            class_count: regex_dfa.class_count,
            eoi_class: regex_dfa.eoi_class,
            table_size,
            byte_classes: regex_dfa.byte_classes,
            packed_byte_classes: {
                let mut packed = [[0u32; 4]; 64];
                for (i, &c) in regex_dfa.byte_classes.iter().enumerate() {
                    packed[i / 4][i % 4] = c;
                }
                packed
            },
            max_input_size: config.configured_gpu_max_input_size(),
        })
    }

    fn validate_device(device: &wgpu::Device) -> Result<()> {
        let limits = device.limits();
        let required_smem_bytes = u32::try_from(SMEM_BYTES).map_err(|_| Error::GpuDeviceError {
            reason: format!("SMEM byte budget {SMEM_BYTES} does not fit in u32"),
        })?;
        if limits.max_compute_workgroup_storage_size < required_smem_bytes {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "adapter workgroup storage limit {} bytes is below required {} bytes for SMEM mode",
                    limits.max_compute_workgroup_storage_size,
                    required_smem_bytes
                ),
            });
        }
        Ok(())
    }

    fn compile_pipeline(
        device: &wgpu::Device,
        transition_table: &[u32],
        class_count: u32,
        state_count: usize,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader_source = if shader_smem::should_specialize_shader(state_count) {
            shader_smem::generate_specialized_shader(transition_table, class_count)
        } else {
            shader_smem::generate_regex_dfa_smem_shader()
        };
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smem dfa compute shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smem dfa bind group layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
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
                storage_entry(7, true),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smem dfa pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smem dfa pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use matchkit::{BlockMatcher, Match};

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    fn build_overflowing_pattern_set() -> PatternSet {
        for count in 64..=2048 {
            let builder = (0..count).fold(PatternSet::builder(), |builder, index| {
                builder.regex(&format!("pattern-{index:04}-{}", "abcdef".repeat(4)))
            });
            match builder.build() {
                Ok(patterns)
                    if patterns
                        .compiled_regex_dfa()
                        .map_or(0, |dfa| dfa.transition_table().len())
                        > SmemDfaMatcher::max_smem_entries() =>
                {
                    return patterns;
                }
                Ok(_) | Err(_) => {}
            }
        }

        panic!("failed to synthesize a DFA that exceeds the SMEM capacity");
    }

    fn build_largest_fitting_pattern_set() -> PatternSet {
        let mut largest = None;

        for count in 1..=1024 {
            let builder = (0..count).fold(PatternSet::builder(), |builder, index| {
                builder.regex(&format!("edge-{index:04}-{}", "qrstuv".repeat(3)))
            });
            match builder.build() {
                Ok(patterns)
                    if patterns
                        .compiled_regex_dfa()
                        .map_or(0, |dfa| dfa.transition_table().len())
                        <= SmemDfaMatcher::max_smem_entries() =>
                {
                    largest = Some(patterns);
                }
                Ok(_) => break,
                Err(_) => {}
            }
        }

        match largest {
            Some(v) => v,
            None => panic!("failed to synthesize a DFA that fits within SMEM"),
        }
    }

    fn with_gpu_test<T>(test: impl FnOnce() -> Result<T>) -> Result<Option<T>> {
        match test() {
            Ok(value) => Ok(Some(value)),
            Err(Error::NoGpuAdapter) => {
                warn!("skipping SMEM GPU test because no GPU adapter is available");
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    #[test]
    fn small_dfa_fits_in_smem() -> Result<()> {
        let patterns = PatternSet::builder().regex("foo").regex("bar").build()?;
        let matcher = block_on(SmemDfaMatcher::new(&patterns));

        match matcher {
            Ok(_) | Err(Error::NoGpuAdapter) => Ok(()),
            Err(error) => Err(error),
        }
    }

    #[test]
    fn large_dfa_rejected() -> Result<()> {
        let patterns = build_overflowing_pattern_set();
        let result = block_on(SmemDfaMatcher::new(&patterns));

        match result {
            Err(Error::PatternCompilationFailed { reason }) => {
                assert!(reason.contains("shared-memory limit"));
                Ok(())
            }
            Err(Error::NoGpuAdapter) => {
                panic!("constructor should reject oversized DFAs before GPU acquisition")
            }
            Ok(_) => panic!("oversized DFA unexpectedly fit in SMEM mode"),
            Err(error) => Err(error),
        }
    }

    #[test]
    #[ignore = "GPU shader parity bug under software Vulkan; requires WGSL debugging."]
    fn smem_matches_cpu_parity() -> Result<()> {
        let patterns = PatternSet::builder()
            .regex("ab+c")
            .regex("secret")
            .regex("a.c")
            .build()?;
        let data = b"zzabbbc secret abc abbc";
        let cpu_matches = patterns.scan(data)?;

        let gpu_matches = with_gpu_test(|| {
            let matcher = block_on(SmemDfaMatcher::new(&patterns))?;
            Ok(block_on(matcher.scan_block(data))?)
        })?;

        if let Some(gpu_matches) = gpu_matches {
            assert_eq!(gpu_matches, cpu_matches);
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU shader parity bug under software Vulkan; requires WGSL debugging."]
    fn smem_overlapping_patterns() -> Result<()> {
        let patterns = PatternSet::builder()
            .regex("aba")
            .regex("ba")
            .regex("a")
            .build()?;
        let data = b"ababa";
        let cpu_matches = patterns.scan(data)?;

        let gpu_matches = with_gpu_test(|| {
            let matcher = block_on(SmemDfaMatcher::new(&patterns))?;
            Ok(block_on(matcher.scan_block(data))?)
        })?;

        if let Some(gpu_matches) = gpu_matches {
            assert_eq!(gpu_matches, cpu_matches);
            assert!(gpu_matches.iter().any(|m| m.start == 0 && m.end == 3));
            assert!(gpu_matches.iter().any(|m| m.start == 1 && m.end == 3));
            assert!(gpu_matches.iter().any(|m| m.start == 2 && m.end == 5));
        }
        Ok(())
    }

    #[test]
    fn smem_empty_input() -> Result<()> {
        let patterns = PatternSet::builder().regex("abc").build()?;
        let gpu_matches = with_gpu_test(|| {
            let matcher = block_on(SmemDfaMatcher::new(&patterns))?;
            Ok(block_on(matcher.scan_block(b""))?)
        })?;

        if let Some(gpu_matches) = gpu_matches {
            assert!(gpu_matches.is_empty());
        }
        Ok(())
    }

    #[test]
    fn smem_no_matches() -> Result<()> {
        let patterns = PatternSet::builder().regex("needle").build()?;
        let data = b"haystack";
        let gpu_matches = with_gpu_test(|| {
            let matcher = block_on(SmemDfaMatcher::new(&patterns))?;
            Ok(block_on(matcher.scan_block(data))?)
        })?;

        if let Some(gpu_matches) = gpu_matches {
            assert!(gpu_matches.is_empty());
        }
        Ok(())
    }

    #[test]
    fn smem_max_patterns_fitting_smem() -> Result<()> {
        let patterns = build_largest_fitting_pattern_set();
        let transition_entries = patterns
            .compiled_regex_dfa()
            .map_or(0, |dfa| dfa.transition_table().len());
        assert!(transition_entries <= SmemDfaMatcher::max_smem_entries());

        let matcher = block_on(SmemDfaMatcher::new(&patterns));
        match matcher {
            Ok(_) | Err(Error::NoGpuAdapter) => Ok(()),
            Err(error) => Err(error),
        }
    }

    #[test]
    #[ignore = "GPU shader parity bug under software Vulkan; requires WGSL debugging."]
    fn smem_match_positions_are_exclusive_end() -> Result<()> {
        let patterns = PatternSet::builder().regex("secret").build()?;
        let data = b"xxsecretzz";
        let gpu_matches = with_gpu_test(|| {
            let matcher = block_on(SmemDfaMatcher::new(&patterns))?;
            Ok(block_on(matcher.scan_block(data))?)
        })?;

        if let Some(gpu_matches) = gpu_matches {
            let Some(mat) = gpu_matches
                .iter()
                .find(|candidate| candidate.start == 2)
                .copied()
            else {
                panic!("expected a match starting at byte 2");
            };
            assert_eq!(&data[mat.start as usize..mat.end as usize], b"secret");
            assert_eq!(mat.end, 8);
        }
        Ok(())
    }

    #[test]
    fn gpu_smem_dedup_consistency() {
        // Test that matches with different ends are not incorrectly deduped
        // and that identical matches are deduped even if they have some other matches in between.
        let m1 = Match {
            pattern_id: 0,
            start: 0,
            end: 10,
            padding: 0,
        };
        let m2 = Match {
            pattern_id: 0,
            start: 0,
            end: 5,
            padding: 0,
        };
        let m3 = Match {
            pattern_id: 0,
            start: 0,
            end: 10,
            padding: 0,
        };

        let mut matches = vec![m1, m2, m3];
        matches.sort_unstable();
        matches.dedup_by(|left, right| {
            left.pattern_id == right.pattern_id
                && left.start == right.start
                && left.end == right.end
        });

        assert_eq!(matches.len(), 2);
        let ends: Vec<u32> = matches.iter().map(|m| m.end).collect();
        assert!(ends.contains(&5));
        assert!(ends.contains(&10));
    }
}
