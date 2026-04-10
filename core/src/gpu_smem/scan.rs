use crate::gpu_smem::state::{PipelineState, Uniforms, COUNT_BUFFER_U32S, MAX_MATCHES};
use crate::gpu_smem::SmemDfaMatcher;
use crate::shader::WORKGROUP_SIZE;
use crate::shader_smem;
use crate::{
    error::{Error, Result},
    GpuMatch, Match,
};

/// Maximum scan depth per thread to prevent O(N²) GPU TDR on large inputs.
///
/// Each thread starts at a different position and scans forward. Without a cap,
/// a 128MB input results in ~500K threads each scanning up to 128MB bytes,
/// which causes GPU timeout/reset. The same cap is applied in the non-SMEM
/// shader (see `src/shader.rs`).
const DEFAULT_MAX_SCAN_DEPTH: u32 = 16_777_216;
use matchkit::BlockMatcher;
use tracing::warn;

impl SmemDfaMatcher {
    pub(super) fn upload_buffers(&self, state: &PipelineState, data: &[u8], input_len: u32) {
        let aligned_len = data.len() & !3;
        if aligned_len > 0 {
            self.queue
                .write_buffer(&state.input_buf, 0, &data[..aligned_len]);
        }
        let remainder = data.len() - aligned_len;
        if remainder > 0 {
            let mut tail = [0u8; 4];
            tail[..remainder].copy_from_slice(&data[aligned_len..]);
            self.queue
                .write_buffer(&state.input_buf, aligned_len as u64, &tail);
        }

        self.queue.write_buffer(
            &state.count_buf,
            0,
            bytemuck::cast_slice(&[0u32, 0u32, shader_smem::STATUS_OK]),
        );

        let uniforms = Uniforms {
            input_len,
            start_state: self.start_state,
            max_matches: MAX_MATCHES,
            class_count: self.class_count,
            max_scan_depth: input_len.min(DEFAULT_MAX_SCAN_DEPTH),
            eoi_class: self.eoi_class,
            table_size: self.table_size,
            _padding0: 0,
            byte_classes: self.packed_byte_classes,
        };

        self.queue
            .write_buffer(&state.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub(super) fn dispatch_shader(&self, state: &PipelineState, input_len: u32) {
        let total_workgroups = input_len.div_ceil(WORKGROUP_SIZE);
        let max_x = 65_535;
        let workgroups_x = total_workgroups.min(max_x);
        let workgroups_y = total_workgroups.div_ceil(max_x);
        let match_buf_size = u64::from(MAX_MATCHES) * std::mem::size_of::<GpuMatch>() as u64;
        let count_buf_size = COUNT_BUFFER_U32S * std::mem::size_of::<u32>() as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("smem dfa encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smem dfa pass"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &state.bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&state.count_buf, 0, &state.count_staging, 0, count_buf_size);
        encoder.copy_buffer_to_buffer(&state.match_buf, 0, &state.match_staging, 0, match_buf_size);
        let _ = self.queue.submit(Some(encoder.finish()));
    }

    pub(super) async fn readback_results(&self, state: &PipelineState) -> Result<Vec<Match>> {
        let (tx_count, rx_count) = std::sync::mpsc::channel();
        state.count_staging.slice(..).map_async(
            wgpu::MapMode::Read,
            move |result: std::result::Result<(), wgpu::BufferAsyncError>| {
                let _ = tx_count.send(result);
            },
        );

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            if let Ok(result) = rx_count.try_recv() {
                if result.is_err() {
                    return Err(Error::BufferMapFailed);
                }
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::GpuDeviceError {
                    reason: "SMEM DFA count buffer map timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let count_data = state.count_staging.slice(..).get_mapped_range();
        let counts: &[u32] = bytemuck::cast_slice(&count_data);
        let match_count = counts.first().copied().unwrap_or_default() as usize;
        let overflow: u32 = counts.get(1).copied().unwrap_or_default();
        let status = match counts.get(2).copied() {
            Some(v) => v,
            None => shader_smem::STATUS_TABLE_TOO_LARGE,
        };
        drop(count_data);
        state.count_staging.unmap();

        if status == shader_smem::STATUS_TABLE_TOO_LARGE {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "DFA transition table exceeded SMEM capacity at runtime: {} entries > {} entries",
                    self.table_size,
                    Self::max_smem_entries()
                ),
            });
        }

        if overflow != 0 {
            return Err(Error::MatchBufferOverflow {
                count: match_count.saturating_add(overflow as usize),
                max: MAX_MATCHES as usize,
            });
        }
        if match_count == 0 {
            return Ok(Vec::new());
        }

        let (tx_match, rx_match) = std::sync::mpsc::channel();
        state.match_staging.slice(..).map_async(
            wgpu::MapMode::Read,
            move |result: std::result::Result<(), wgpu::BufferAsyncError>| {
                let _ = tx_match.send(result);
            },
        );

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            if let Ok(result) = rx_match.try_recv() {
                if result.is_err() {
                    return Err(Error::BufferMapFailed);
                }
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::GpuDeviceError {
                    reason: "SMEM DFA match buffer map timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let match_data = state.match_staging.slice(..).get_mapped_range();
        let raw_matches: &[GpuMatch] = bytemuck::cast_slice(&match_data);
        let valid_count = match_count.min(MAX_MATCHES as usize);
        let mut matches: Vec<Match> = raw_matches[..valid_count]
            .iter()
            .copied()
            .map(Match::from)
            .collect();
        drop(match_data);
        state.match_staging.unmap();

        matches.sort_unstable();
        matches.dedup_by(|left, right| {
            left.pattern_id == right.pattern_id
                && left.start == right.start
                && left.end == right.end
        });
        Ok(canonicalize_regex_matches(matches))
    }

    pub(super) async fn scan_block_impl(&self, data: &[u8]) -> Result<Vec<Match>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        if data.len() > self.max_input_size {
            return Err(Error::InputTooLarge {
                bytes: data.len(),
                max_bytes: self.max_input_size,
            });
        }

        let input_len = u32::try_from(data.len()).map_err(|_| Error::InputTooLarge {
            bytes: data.len(),
            max_bytes: u32::MAX as usize,
        })?;

        let state_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        let state = loop {
            if let Some(state) = self.states.pop() {
                break state;
            }
            if tokio::time::Instant::now() >= state_deadline {
                return Err(Error::GpuDeviceError {
                    reason: "SMEM DFA pipeline state pool exhausted (30s timeout)".to_string(),
                });
            }
            tokio::task::yield_now().await;
        };

        self.upload_buffers(&state, data, input_len);
        self.dispatch_shader(&state, input_len);
        let result = self.readback_results(&state).await;
        let pushed = self.states.push(state);
        if pushed.is_err() {
            warn!("pipeline state pool unexpectedly full after SMEM scan");
        }
        result
    }
}

fn canonicalize_regex_matches(mut matches: Vec<Match>) -> Vec<Match> {
    if matches.is_empty() {
        return matches;
    }

    matches.sort_unstable_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then_with(|| right.end.cmp(&left.end))
            .then_with(|| left.pattern_id.cmp(&right.pattern_id))
    });

    let mut canonical = Vec::with_capacity(matches.len());
    let mut index = 0usize;
    let mut cursor_end = 0u32;

    while index < matches.len() {
        let start = matches[index].start;
        if start < cursor_end {
            index += 1;
            continue;
        }

        let mut max_end = matches[index].end;
        let mut group_end = index + 1;
        while group_end < matches.len() && matches[group_end].start == start {
            max_end = max_end.max(matches[group_end].end);
            group_end += 1;
        }

        for candidate in &matches[index..group_end] {
            if candidate.end == max_end {
                canonical.push(*candidate);
            }
        }

        cursor_end = max_end;
        index = group_end;
    }

    canonical
}

#[async_trait::async_trait]
impl BlockMatcher for SmemDfaMatcher {
    async fn scan_block(&self, data: &[u8]) -> matchkit::Result<Vec<Match>> {
        self.scan_block_impl(data).await.map_err(Into::into)
    }

    fn max_block_size(&self) -> usize {
        self.max_input_size
    }
}
