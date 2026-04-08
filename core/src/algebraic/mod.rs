//! Algebraic parallel DFA execution via GPU prefix scan.

pub(crate) mod pipeline;
pub mod readback;
pub mod scan;
pub mod shader;
#[cfg(test)]
mod tests;

use crate::dfa::RegexDFA;
use crate::error::{Error, Result};
use crate::gpu::SharedDeviceQueue;
use crate::matcher::BlockMatcher;
use crate::pattern::PatternSet;
use crate::Match;
use std::sync::Mutex;

pub(crate) use pipeline::AlgebraicState;
use pipeline::{Pipelines, StaticResources};

/// Preferred DFA state budget for the pure algebraic fast path.
///
/// At 64 states, the composition table per element is 64^2 × 4B = 16KB — well
/// within GPU per-thread register budgets. Higher state counts (128+) cause
/// register spilling on some GPUs, producing false negatives. The hybrid path
/// handles DFAs above this limit by combining algebraic prefix scan with
/// sequential tail processing.
///
/// Use [`MAX_ALGEBRAIC_STATES_EXTENDED`] for high-end GPUs (5090+) that can
/// handle larger composition tables.
pub const MAX_ALGEBRAIC_STATES: u32 = 64;

/// Extended algebraic state limit for high-end GPUs (RTX 4090/5090+).
///
/// On GPUs with 48KB+ shared memory and high register counts, 128-state
/// composition tables (128^2 × 4B = 64KB global memory) work correctly.
/// Enable via `AutoMatcherConfig::algebraic_state_limit(128)`.
pub const MAX_ALGEBRAIC_STATES_EXTENDED: u32 = 128;
pub(crate) const HYBRID_ESCAPE_STATE: u32 = MAX_ALGEBRAIC_STATES;
pub(crate) const HYBRID_DEAD_STATE: u32 = MAX_ALGEBRAIC_STATES + 1;
pub(crate) const HYBRID_GPU_STATES: u32 = MAX_ALGEBRAIC_STATES + 2;

/// Previous algebraic limit (64 states). Used for backward compat in tests.
#[cfg(test)]
pub(crate) const PREVIOUS_MAX_STATES: u32 = 64;

pub(crate) const DEFAULT_BLOCK_SIZE: usize = 4096;
pub(crate) const DEFAULT_MAX_INPUT_SIZE: usize = 256 * 1024 * 1024;
pub(crate) const MAX_MATCHES: u32 = 1_048_576;
pub(crate) const MAX_SCAN_ROUNDS: usize = 12;

/// Algebraic parallel DFA execution via prefix scan.
///
/// Small DFAs stay on the pure algebraic fast path. Larger DFAs use a hybrid
/// path that algebraically scans the first 32-state chunk and then continues
/// sequentially once execution escapes that chunk.
#[derive(Debug)]
pub struct AlgebraicDfaMatcher {
    pub(crate) dfa: RegexDFA,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) map_pipeline: wgpu::ComputePipeline,
    pub(crate) scan_pipeline: wgpu::ComputePipeline,
    pub(crate) extract_pipeline: wgpu::ComputePipeline,
    pub(crate) gpu_state_count: u32,
    pub(crate) full_state_count: u32,
    pub(crate) start_state: u32,
    /// Pre-packed byte classes for GPU uniform upload. Computed once at
    /// construction instead of per-block. At 1TB scale with 4KB blocks,
    /// this eliminates 262,144 × 256 iterations = 67M wasted ops.
    pub(crate) packed_byte_classes: [[u32; 4]; 64],
    pub(crate) byte_classes: [u32; 256],
    pub(crate) class_count: u32,
    pub(crate) eoi_class: u32,
    pub(crate) max_input_size: usize,
    pub(crate) block_size: usize,
    pub(crate) state: Mutex<Option<AlgebraicState>>,
}

impl AlgebraicDfaMatcher {
    /// Compile a DFA into the algebraic GPU backend.
    pub async fn new(patterns: &PatternSet) -> Result<Self> {
        let device_queue = crate::gpu::acquire_device().await?;
        Self::from_device(device_queue, patterns)
    }

    /// Compile a DFA into the algebraic GPU backend using an existing device.
    pub fn from_device(device_queue: SharedDeviceQueue, patterns: &PatternSet) -> Result<Self> {
        let regex_dfa = patterns.compiled_regex_dfa()?.into_owned();
        let state_count = u32::try_from(regex_dfa.state_count()).map_err(|_| {
            Error::PatternCompilationFailed {
                reason: "dfa state count does not fit in u32".to_string(),
            }
        })?;
        let gpu_state_count = if state_count > MAX_ALGEBRAIC_STATES {
            HYBRID_GPU_STATES
        } else {
            state_count
        };

        let device = device_queue.0.clone();
        let queue = device_queue.1.clone();
        let resources = if state_count > MAX_ALGEBRAIC_STATES {
            StaticResources::new_hybrid(&device, &regex_dfa)
        } else {
            StaticResources::new(&device, &regex_dfa, regex_dfa.pattern_lengths())
        };
        let pipelines = Pipelines::new(&device);
        let state = AlgebraicState::new(
            &device,
            &resources,
            gpu_state_count,
            DEFAULT_BLOCK_SIZE,
            &pipelines.map_layout,
            &pipelines.scan_layout,
            &pipelines.extract_layout,
        );

        Ok(Self {
            dfa: regex_dfa.clone(),
            device,
            queue,
            map_pipeline: pipelines.map_pipeline,
            scan_pipeline: pipelines.scan_pipeline,
            extract_pipeline: pipelines.extract_pipeline,
            gpu_state_count,
            full_state_count: state_count,
            start_state: regex_dfa.start_state & 0x3FFF_FFFF,
            byte_classes: regex_dfa.byte_classes,
            packed_byte_classes: {
                let mut packed = [[0u32; 4]; 64];
                for (i, &c) in regex_dfa.byte_classes.iter().enumerate() {
                    packed[i / 4][i % 4] = c;
                }
                packed
            },
            class_count: regex_dfa.class_count,
            eoi_class: regex_dfa.eoi_class,
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            block_size: DEFAULT_BLOCK_SIZE,
            state: Mutex::new(Some(state)),
        })
    }

    pub(crate) async fn take_state(&self) -> Result<AlgebraicState> {
        let mut backoff_us = 10_u64;
        loop {
            let state = {
                let mut guard = self.state.lock().map_err(|_| Error::GpuDeviceError {
                    reason: "algebraic mutex poisoned — prior scan panicked, state may be corrupt"
                        .into(),
                })?;
                guard.take()
            };

            if let Some(state) = state {
                return Ok(state);
            }
            tokio::time::sleep(std::time::Duration::from_micros(backoff_us)).await;
            backoff_us = (backoff_us * 2).min(10_000);
        }
    }

    pub(crate) fn put_state(&self, state: AlgebraicState) {
        // If the mutex is poisoned, we still put the state back to allow
        // recovery on the next take_state call. The state we're putting
        // back is freshly produced by a successful scan, so it's valid.
        let mut guard = match self.state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = Some(state);
    }
}

impl BlockMatcher for AlgebraicDfaMatcher {
    async fn scan_block(&self, data: &[u8]) -> matchkit::Result<Vec<Match>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data.len() > self.max_input_size {
            return Err(Error::InputTooLarge {
                bytes: data.len(),
                max_bytes: self.max_input_size,
            }
            .into());
        }
        if self.full_state_count > MAX_ALGEBRAIC_STATES {
            return self.scan_hybrid(data).await.map_err(Into::into);
        }

        let state = self.take_state().await.map_err(matchkit::Error::from)?;
        let mut matches = Vec::new();
        let mut carry_state = self.start_state;
        let mut offset = 0;

        while offset < data.len() {
            let block_len = std::cmp::min(data.len() - offset, self.block_size);
            let chunk = &data[offset..offset + block_len];

            self.upload_input(&state, chunk);
            let round_count = self
                .upload_uniforms(&state, block_len, carry_state, offset)
                .map_err(Into::<matchkit::Error>::into)?;
            self.dispatch_block(&state, block_len, round_count, false);

            let mut block_matches = self
                .read_matches(&state)
                .await
                .map_err(Into::<matchkit::Error>::into)?;
            matches.append(&mut block_matches);

            carry_state = self
                .read_tail_state(&state, carry_state)
                .await
                .map_err(Into::<matchkit::Error>::into)?;
            offset += block_len;
        }

        self.put_state(state);
        Ok(matches)
    }

    fn max_block_size(&self) -> usize {
        self.max_input_size
    }
}
