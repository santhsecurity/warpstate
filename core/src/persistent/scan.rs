use crate::dma::DmaStagingBuffer;
use crate::error::{Error, Result};
use crate::matcher::{BlockMatcher, ZeroCopyBlockMatcher};
use crate::persistent::{state, PendingWork, PersistentMatcher, Uniforms};
use crate::{shader, GpuMatch, Match};

/// RAII guard that releases a buffer set on drop, preventing permanent
/// deadlock if a panic occurs between wait_for_buffer_set and release.
struct BufferSetGuard<'a> {
    matcher: &'a PersistentMatcher,
    set_idx: usize,
    released: bool,
}

impl<'a> BufferSetGuard<'a> {
    fn new(matcher: &'a PersistentMatcher, set_idx: usize) -> Self {
        Self {
            matcher,
            set_idx,
            released: false,
        }
    }

    fn release(&mut self) {
        if !self.released {
            self.released = true;
            self.matcher.release_buffer_set(self.set_idx);
        }
    }
}

impl Drop for BufferSetGuard<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

impl BlockMatcher for PersistentMatcher {
    async fn scan_block(&self, data: &[u8]) -> matchkit::Result<Vec<Match>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data.len() > self.max_input_size {
            return Err(matchkit::Error::InputTooLarge {
                bytes: data.len(),
                max_bytes: self.max_input_size,
            });
        }
        // Safe u32 conversion — max_input_size should be clamped to u32::MAX.
        let input_len = u32::try_from(data.len()).map_err(|_| matchkit::Error::InputTooLarge {
            bytes: data.len(),
            max_bytes: u32::MAX as usize,
        })?;
        let set_idx = self.wait_for_buffer_set().await;
        let mut guard = BufferSetGuard::new(self, set_idx);
        self.upload_to_set(set_idx, data, input_len);
        self.dispatch_set(set_idx, input_len);
        let result = self.readback_from_set(set_idx).await.map_err(Into::into);
        guard.release();
        result
    }

    fn max_block_size(&self) -> usize {
        self.max_input_size
    }
}

impl ZeroCopyBlockMatcher for PersistentMatcher {
    fn create_staging_buffer(&self, input_len: usize) -> Result<DmaStagingBuffer> {
        if input_len > self.max_input_size {
            return Err(Error::InputTooLarge {
                bytes: input_len,
                max_bytes: self.max_input_size,
            });
        }
        let padded_len =
            input_len
                .max(1)
                .checked_next_multiple_of(4)
                .ok_or(Error::InputTooLarge {
                    bytes: input_len,
                    max_bytes: usize::MAX - 3,
                })?;
        DmaStagingBuffer::with_lengths(&self.device, padded_len, input_len)
    }

    async fn scan_zero_copy_block(
        &self,
        staging: DmaStagingBuffer,
        input_len: usize,
    ) -> Result<Vec<Match>> {
        if input_len == 0 {
            return Ok(Vec::new());
        }
        // Validate input_len fits u32 and matches staging capacity.
        let input_len_u32 = u32::try_from(input_len).map_err(|_| Error::InputTooLarge {
            bytes: input_len,
            max_bytes: u32::MAX as usize,
        })?;
        let set_idx = self.wait_for_buffer_set().await;
        let mut guard = BufferSetGuard::new(self, set_idx);
        self.upload_staging_to_set(set_idx, staging, input_len_u32);
        self.dispatch_set(set_idx, input_len_u32);
        let result = self.readback_from_set(set_idx).await;
        guard.release();
        result
    }
}

impl PersistentMatcher {
    pub(crate) fn upload_to_set(&self, set_idx: usize, data: &[u8], input_len: u32) {
        let buffer_set = &self.buffer_sets[set_idx];

        let aligned_len = data.len() & !3;
        if aligned_len > 0 {
            self.queue
                .write_buffer(&buffer_set.input_buf, 0, &data[..aligned_len]);
        }

        let remainder = data.len() - aligned_len;
        if remainder > 0 {
            let mut padded_tail = [0_u8; 4];
            padded_tail[..remainder].copy_from_slice(&data[aligned_len..]);
            self.queue
                .write_buffer(&buffer_set.input_buf, aligned_len as u64, &padded_tail);
        }

        self.prepare_set(set_idx, input_len);
    }

    pub(crate) fn upload_staging_to_set(
        &self,
        set_idx: usize,
        mut staging: DmaStagingBuffer,
        input_len: u32,
    ) {
        self.prepare_set(set_idx, input_len);
        if staging.capacity() == 0 {
            return;
        }

        staging.flush_to_vram();
        let buffer_set = &self.buffer_sets[set_idx];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("persistent zero-copy upload"),
            });
        encoder.copy_buffer_to_buffer(
            staging.buffer(),
            0,
            &buffer_set.input_buf,
            0,
            staging.capacity() as u64,
        );
        self.queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn prepare_set(&self, set_idx: usize, input_len: u32) {
        let buffer_set = &self.buffer_sets[set_idx];

        self.queue.write_buffer(
            &buffer_set.count_buf,
            0,
            bytemuck::cast_slice(&[0_u32, 0_u32]),
        );

        let uniforms = Uniforms {
            input_len,
            start_state: self.start_state,
            max_matches: self.max_matches,
            class_count: self.class_count,
            eoi_class: self.eoi_class,
            padding: [0; 3],
            byte_classes: super::pack_byte_classes(&self.byte_classes),
        };

        self.queue
            .write_buffer(&buffer_set.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut state = state::lock_state(&self.state);
        state.pending = Some(PendingWork {
            buffer_set_idx: set_idx,
            input_len,
        });
    }

    pub(crate) fn dispatch_set(&self, set_idx: usize, input_len: u32) {
        let total_workgroups = input_len.div_ceil(shader::WORKGROUP_SIZE);
        let max_x = 65_535_u32;
        let workgroups_x = total_workgroups.min(max_x);
        let workgroups_y = total_workgroups.div_ceil(max_x);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("persistent matcher encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("persistent matcher pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[set_idx], &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let buffer_set = &self.buffer_sets[set_idx];
        let match_buf_size = super::match_buffer_size_bytes(self.max_matches);
        encoder.copy_buffer_to_buffer(&buffer_set.count_buf, 0, &buffer_set.count_staging, 0, 8);
        encoder.copy_buffer_to_buffer(
            &buffer_set.match_buf,
            0,
            &buffer_set.match_staging,
            0,
            match_buf_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let mut state = state::lock_state(&self.state);
        if state.pending.as_ref().is_some_and(|pending| {
            pending.buffer_set_idx == set_idx && pending.input_len == input_len
        }) {
            state.pending = None;
        }
    }

    pub(crate) async fn readback_from_set(&self, set_idx: usize) -> Result<Vec<Match>> {
        let buffer_set = &self.buffer_sets[set_idx];

        let (tx_count, rx_count) = std::sync::mpsc::channel();
        buffer_set.count_staging.slice(..).map_async(
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
                    reason: "persistent matcher count readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let count_data = buffer_set.count_staging.slice(..).get_mapped_range();
        let counts: &[u32] = bytemuck::cast_slice(&count_data);
        if counts.len() < 2 {
            drop(count_data);
            buffer_set.count_staging.unmap();
            return Err(Error::GpuDeviceError {
                reason: "persistent matcher count buffer too small".to_string(),
            });
        }
        let match_count = counts[0] as usize;
        let overflow_count = counts[1] as usize;
        drop(count_data);
        buffer_set.count_staging.unmap();

        if overflow_count != 0 {
            return Err(Error::MatchBufferOverflow {
                count: match_count + overflow_count,
                max: self.max_matches as usize,
            });
        }

        if match_count == 0 {
            return Ok(Vec::new());
        }

        let (tx_match, rx_match) = std::sync::mpsc::channel();
        buffer_set.match_staging.slice(..).map_async(
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
                    reason: "persistent matcher match readback timed out after 30s".to_string(),
                });
            }
            tokio::task::yield_now().await;
        }

        let match_data = buffer_set.match_staging.slice(..).get_mapped_range();
        let valid_count = match_count.min(self.max_matches as usize);
        let raw_matches: &[GpuMatch] = bytemuck::cast_slice(&match_data);
        let mut matches: Vec<Match> = raw_matches[..valid_count]
            .iter()
            .copied()
            .map(Match::from)
            .collect();
        drop(match_data);
        buffer_set.match_staging.unmap();

        matches.sort_unstable();
        matches.dedup_by(|left, right| {
            left.pattern_id == right.pattern_id
                && left.start == right.start
                && left.end == right.end
        });
        Ok(matches)
    }
}
