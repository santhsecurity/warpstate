pub(crate) use crate::algebraic::pipeline::AlgebraicState;
use crate::algebraic::{AlgebraicDfaMatcher, MAX_MATCHES};
use crate::error::{Error, Result};
use crate::{GpuMatch, Match};

impl AlgebraicDfaMatcher {
    pub(crate) async fn read_count(&self, state: &AlgebraicState) -> Result<(usize, bool)> {
        let (tx, rx) = std::sync::mpsc::channel();
        state.count_staging.slice(..).map_async(
            wgpu::MapMode::Read,
            move |result: std::result::Result<(), wgpu::BufferAsyncError>| {
                let _ = tx.send(result);
            },
        );

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
                    reason: "algebraic DFA count readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let count_data = state.count_staging.slice(..).get_mapped_range();
        let counts: &[u32] = bytemuck::cast_slice(&count_data);
        if counts.len() < 2 {
            drop(count_data);
            state.count_staging.unmap();
            return Err(Error::GpuDeviceError {
                reason: "algebraic DFA count buffer too small".to_string(),
            });
        }
        let match_count = counts[0] as usize;
        let overflow = counts[1] != 0;
        drop(count_data);
        state.count_staging.unmap();
        Ok((match_count, overflow))
    }

    pub(crate) async fn read_matches(&self, state: &AlgebraicState) -> Result<Vec<Match>> {
        let (match_count, overflow) = self.read_count(state).await?;
        if overflow {
            return Err(Error::MatchBufferOverflow {
                count: match_count,
                max: MAX_MATCHES as usize,
            });
        }
        if match_count == 0 {
            return Ok(Vec::new());
        }

        let (tx, rx) = std::sync::mpsc::channel();
        state.match_staging.slice(..).map_async(
            wgpu::MapMode::Read,
            move |result: std::result::Result<(), wgpu::BufferAsyncError>| {
                let _ = tx.send(result);
            },
        );

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
                    reason: "algebraic DFA match readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let match_data = state.match_staging.slice(..).get_mapped_range();
        let raw: &[GpuMatch] = bytemuck::cast_slice(&match_data);
        let valid = std::cmp::min(match_count, MAX_MATCHES as usize);
        let matches = raw[..valid].iter().copied().map(Match::from).collect();
        drop(match_data);
        state.match_staging.unmap();
        Ok(matches)
    }

    pub(crate) async fn read_tail_state(
        &self,
        state: &AlgebraicState,
        carry_state: u32,
    ) -> Result<u32> {
        let tail_size = (u64::from(self.gpu_state_count) * std::mem::size_of::<u32>() as u64)
            .next_multiple_of(8);
        let (tx, rx) = std::sync::mpsc::channel();
        state
            .tail_staging
            .slice(..tail_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
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
                    reason: "algebraic DFA tail readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let tail_data = state.tail_staging.slice(..tail_size).get_mapped_range();
        let tail: &[u32] = bytemuck::cast_slice(&tail_data);

        // Validate carry_state is valid for indexing
        let index = usize::try_from(carry_state).map_err(|_| Error::PatternCompilationFailed {
            reason: "carry state does not fit in usize".to_string(),
        })?;

        // Additional validation: carry_state must be within GPU state count
        if index >= self.gpu_state_count as usize {
            drop(tail_data);
            state.tail_staging.unmap();
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "carry state {} exceeds GPU state count {} (full state count: {})",
                    carry_state, self.gpu_state_count, self.full_state_count
                ),
            });
        }

        let next_state = *tail.get(index).ok_or_else(|| {
            Error::PatternCompilationFailed {
                reason: format!(
                    "tail buffer index {index} out of bounds (tail has {} entries, gpu_state_count: {})",
                    tail.len(), self.gpu_state_count
                ),
            }
        })?;
        drop(tail_data);
        state.tail_staging.unmap();
        Ok(next_state)
    }

    pub(crate) async fn read_prefix_states(
        &self,
        state: &AlgebraicState,
        block_len: usize,
        carry_state: u32,
    ) -> Result<Vec<u32>> {
        // Validate block_len to prevent overflow in byte_len calculation
        let max_safe_block_len = (u64::MAX as usize)
            .saturating_div(self.gpu_state_count as usize)
            .saturating_div(std::mem::size_of::<u32>());
        if block_len > max_safe_block_len {
            return Err(Error::InputTooLarge {
                bytes: block_len,
                max_bytes: max_safe_block_len,
            });
        }

        let byte_len =
            (block_len * self.gpu_state_count as usize * std::mem::size_of::<u32>()) as u64;
        let (tx, rx) = std::sync::mpsc::channel();
        state
            .func_staging
            .slice(..byte_len)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
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
                    reason: "algebraic DFA func readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        // Validate carry_state is within GPU state count BEFORE mapping (avoids borrow issues)
        let carry_index =
            usize::try_from(carry_state).map_err(|_| Error::PatternCompilationFailed {
                reason: "carry state does not fit in usize".to_string(),
            })?;
        if carry_index >= self.gpu_state_count as usize {
            state.func_staging.unmap();
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "carry state {} exceeds GPU state count {}",
                    carry_state, self.gpu_state_count
                ),
            });
        }

        // Use a scope to ensure `mapped` is dropped before we call unmap
        let prefix_states = {
            let mapped = state.func_staging.slice(..byte_len).get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);

            // CRITICAL: Bounds check before indexing
            let expected_elements = block_len * self.gpu_state_count as usize;
            if raw.len() < expected_elements {
                return Err(Error::PatternCompilationFailed {
                    reason: format!(
                        "prefix state buffer size mismatch: got {} elements, expected {} (block_len={}, gpu_state_count={})",
                        raw.len(), expected_elements, block_len, self.gpu_state_count
                    ),
                });
            }

            let mut prefix_states = Vec::with_capacity(block_len);
            let stride = self.gpu_state_count as usize;
            for pos in 0..block_len {
                let idx = pos * stride + carry_index;
                // Defensive: this should always be true given the check above, but asserts document the invariant
                debug_assert!(
                    idx < raw.len(),
                    "index {} out of bounds {} at pos {} with stride {} and carry_index {}",
                    idx,
                    raw.len(),
                    pos,
                    stride,
                    carry_index
                );
                prefix_states.push(raw[idx]);
            }
            prefix_states
        }; // `mapped` and `raw` are dropped here

        state.func_staging.unmap();
        Ok(prefix_states)
    }
}
