//! Shared-memory regex ensemble GPU matcher.

use crate::dfa::{RegexDFA, MASK_STATE};
use crate::error::{Error, Result};
use crate::gpu::device;
use crate::gpu::readback;
use wgpu::util::DeviceExt;

const SHARED_MEMORY_BYTES: usize = 48 * 1024;
const SHARED_TABLE_WORDS: usize = SHARED_MEMORY_BYTES / std::mem::size_of::<u32>();
const BYTE_CLASS_WORDS: usize = 64;
const STATE_FLAG_MATCH: u32 = 1;
const STATE_FLAG_DEAD: u32 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DfaMetadata {
    offset_words: u32,
    state_count: u32,
    class_count: u32,
    start_state: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EnsembleUniforms {
    input_length: u32,
    dfa_count: u32,
    _padding0: u32,
    _padding1: u32,
}

/// GPU-ready DFA for one regex.
#[derive(Clone, Debug)]
pub struct SmallDfa {
    /// Flat transition table of `state_count * class_count` entries.
    pub transitions: Vec<u32>,
    /// Number of DFA states.
    pub state_count: u32,
    /// Number of byte classes.
    pub class_count: u32,
    /// Anchored start state.
    pub start_state: u32,
    /// Byte-to-class map used by the DFA.
    pub byte_classes: [u8; 256],
    /// Match state indices.
    pub match_states: Vec<u32>,
    state_flags: Vec<u32>,
}

impl SmallDfa {
    /// Returns `true` when the staged blob fits in 48 KiB of workgroup storage.
    #[must_use]
    pub fn fits_in_shared_memory(&self) -> bool {
        self.blob_words() <= SHARED_TABLE_WORDS
    }

    fn blob_words(&self) -> usize {
        self.transitions.len() + self.state_count as usize + BYTE_CLASS_WORDS
    }

    fn validate(&self) -> Result<()> {
        let expected = self
            .state_count
            .checked_mul(self.class_count)
            .ok_or_else(|| Error::PatternCompilationFailed {
                reason: "small DFA transition dimensions overflowed. Fix: simplify the regex or lower the DFA size limit.".to_string(),
            })? as usize;
        if self.transitions.len() != expected {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "small DFA transition table length {} does not match state_count * class_count {}. Fix: rebuild the regex DFA.",
                    self.transitions.len(),
                    expected
                ),
            });
        }
        for &state in &self.match_states {
            if state >= self.state_count {
                return Err(Error::PatternCompilationFailed {
                    reason: format!(
                        "small DFA match state {state} is out of bounds for state_count {}. Fix: rebuild the regex DFA.",
                        self.state_count
                    ),
                });
            }
        }
        Ok(())
    }

    fn encode_blob(&self, out: &mut Vec<u32>) -> Result<DfaMetadata> {
        self.validate()?;
        let offset_words = u32::try_from(out.len()).map_err(|_| Error::PatternSetTooLarge {
            patterns: 0,
            bytes: out.len().saturating_mul(std::mem::size_of::<u32>()),
            max_bytes: u32::MAX as usize,
        })?;

        out.extend_from_slice(&self.transitions);

        out.extend_from_slice(&self.state_flags);

        for chunk in self.byte_classes.chunks_exact(4) {
            out.push(
                u32::from(chunk[0])
                    | (u32::from(chunk[1]) << 8)
                    | (u32::from(chunk[2]) << 16)
                    | (u32::from(chunk[3]) << 24),
            );
        }

        Ok(DfaMetadata {
            offset_words,
            state_count: self.state_count,
            class_count: self.class_count,
            start_state: self.start_state,
        })
    }

    pub(crate) fn from_regex_dfa(regex_dfa: &RegexDFA) -> Result<Self> {
        let class_count = regex_dfa.class_count();
        let state_count_usize = regex_dfa.state_count();
        let state_count = u32::try_from(state_count_usize).map_err(|_| Error::PatternCompilationFailed {
            reason: format!(
                "regex DFA has {} states, exceeding u32::MAX. Fix: simplify the regex.",
                state_count_usize
            ),
        })?;

        let mut byte_classes = [0u8; 256];
        for (dst, &class_id) in byte_classes.iter_mut().zip(regex_dfa.byte_classes().iter()) {
            *dst = u8::try_from(class_id).map_err(|_| Error::PatternCompilationFailed {
                reason: format!(
                    "regex DFA byte class {class_id} exceeds u8::MAX. Fix: simplify the regex so the byte-class alphabet stays compact."
                ),
            })?;
        }

        let raw_transitions = regex_dfa.transition_table();
        let transitions = raw_transitions
            .iter()
            .map(|entry| entry & MASK_STATE)
            .collect::<Vec<_>>();
        let match_states = regex_dfa
            .match_list_pointers()
            .iter()
            .enumerate()
            .filter_map(|(state, &ptr)| (ptr != 0).then_some(state as u32))
            .collect::<Vec<_>>();
        let mut state_flags = vec![0u32; state_count_usize];
        for &state in &match_states {
            state_flags[state as usize] |= STATE_FLAG_MATCH;
        }
        for state in 0..state_count_usize {
            let start = state * class_count as usize;
            let end = start + class_count as usize;
            let slice = &raw_transitions[start..end];
            if !slice.is_empty()
                && slice.iter().all(|entry| (entry & MASK_STATE) == state as u32)
                && RegexDFA::is_dead_state(slice[0])
            {
                state_flags[state] |= STATE_FLAG_DEAD;
            }
        }

        Ok(Self {
            transitions,
            state_count,
            class_count,
            start_state: regex_dfa.start_state(),
            byte_classes,
            match_states,
            state_flags,
        })
    }
}

/// Compile one regex into an independent DFA for the ensemble backend.
pub fn compile_regex_to_small_dfa(pattern: &str) -> Result<SmallDfa> {
    if pattern.is_empty() {
        return Err(Error::EmptyPattern { index: 0 });
    }
    let regex_dfa = RegexDFA::build_with_limit(&[pattern], &[0], RegexDFA::DEFAULT_DFA_SIZE_LIMIT)?;
    SmallDfa::from_regex_dfa(&regex_dfa)
}

/// Shared-memory GPU regex ensemble matcher.
#[derive(Debug)]
pub struct EnsembleRegexMatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    max_workgroups_per_dimension: u32,
}

impl EnsembleRegexMatcher {
    /// Compile the shared-memory ensemble shader for an existing device.
    #[must_use]
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warpstate regex ensemble shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ensemble_shader.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warpstate regex ensemble layout"),
            entries: &[
                device::storage_entry(0, true),
                device::storage_entry(1, true),
                device::storage_entry(2, true),
                device::storage_entry(3, false),
                device::uniform_entry(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("warpstate regex ensemble pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("warpstate regex ensemble pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
            device,
            queue,
            pipeline,
            layout,
        }
    }

    /// Scan input and return one boolean per DFA indicating whether any match exists.
    pub fn scan(&self, input: &[u8], dfas: &[SmallDfa]) -> Result<Vec<bool>> {
        pollster::block_on(self.scan_async(input, dfas))
    }

    /// Async variant used by the higher-level GPU matcher.
    pub async fn scan_async(&self, input: &[u8], dfas: &[SmallDfa]) -> Result<Vec<bool>> {
        if dfas.is_empty() {
            return Ok(Vec::new());
        }

        let input_length = super::to_u32_len(input.len(), u32::MAX as usize)?;
        let dfa_count = u32::try_from(dfas.len()).map_err(|_| Error::PatternSetTooLarge {
            patterns: dfas.len(),
            bytes: 0,
            max_bytes: u32::MAX as usize,
        })?;

        let packed_input = device::pad_to_u32(input);
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warpstate regex ensemble input"),
            contents: device::packed_u32_as_bytes(&packed_input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut table_words = Vec::new();
        let mut metadata = Vec::with_capacity(dfas.len());
        for dfa in dfas {
            metadata.push(dfa.encode_blob(&mut table_words)?);
        }

        let tables_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warpstate regex ensemble tables"),
            contents: bytemuck::cast_slice(&table_words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let metadata_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warpstate regex ensemble metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = u64::from(dfa_count) * std::mem::size_of::<u32>() as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warpstate regex ensemble output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer =
            device::readback_buffer(&self.device, output_size, "warpstate regex ensemble staging");
        let uniforms = EnsembleUniforms {
            input_length,
            dfa_count,
            _padding0: 0,
            _padding1: 0,
        };
        let uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warpstate regex ensemble uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warpstate regex ensemble bind group"),
            layout: &self.layout,
            entries: &[
                device::entry(0, &input_buffer),
                device::entry(1, &tables_buffer),
                device::entry(2, &metadata_buffer),
                device::entry(3, &output_buffer),
                device::entry(4, &uniform_buffer),
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warpstate regex ensemble encoder"),
        });
        encoder.clear_buffer(&output_buffer, 0, None);

        if dfa_count > self.max_workgroups_per_dimension {
            return Err(Error::PatternSetTooLarge {
                patterns: dfas.len(),
                bytes: table_words.len().saturating_mul(std::mem::size_of::<u32>()),
                max_bytes: self.max_workgroups_per_dimension as usize,
            });
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warpstate regex ensemble pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dfa_count, 1, 1);
        drop(pass);

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        readback::await_buffer_map(
            &self.device,
            &self.queue,
            &staging_buffer,
            "waiting for regex ensemble output buffer",
        )
        .await?;

        let mapped = staging_buffer.slice(..).get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&mapped);
        let result = raw.iter().take(dfas.len()).map(|&value| value != 0).collect();
        drop(mapped);
        staging_buffer.unmap();

        staging_buffer.destroy();
        output_buffer.destroy();
        input_buffer.destroy();
        tables_buffer.destroy();
        metadata_buffer.destroy();
        uniform_buffer.destroy();

        Ok(result)
    }
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn regex_ensemble_scans_100_independent_regexes() -> Result<()> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
            Some(adapter) => adapter,
            None => return Ok(()),
        };
        if crate::gpu::adapter_is_software(&adapter.get_info()) {
            return Ok(());
        }
        let (device, queue) = match block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)) {
            Ok(device_queue) => device_queue,
            Err(_) => return Ok(()),
        };

        let mut dfas = Vec::new();
        let mut expected = Vec::new();
        let mut input = vec![b'x'; 1024 * 1024];

        for index in 0usize..100 {
            let pattern = format!("R{index:03}Z[0-9]{{2}}");
            dfas.push(compile_regex_to_small_dfa(&pattern)?);
            let should_match = index % 3 == 0;
            expected.push(should_match);
            if should_match {
                let offset = index * 7919;
                let needle = format!("R{index:03}Z42");
                input[offset..offset + needle.len()].copy_from_slice(needle.as_bytes());
            }
        }

        let matcher = EnsembleRegexMatcher::new(device, queue);
        let actual = block_on(matcher.scan_async(&input, &dfas))?;
        assert_eq!(actual, expected, "regex ensemble returned incorrect match bitmap");
        Ok(())
    }
}
