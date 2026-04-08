use crate::error::{Error, Result};
use crate::Match;

/// Maximum time to wait for a GPU buffer map before declaring timeout.
const GPU_MAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Wait for both the GPU submission to complete and the buffer map to be ready.
/// This prevents reading zero-filled staging buffers when the device is lost
/// or the shader never executed.
pub(crate) async fn await_buffer_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    timeout_msg: &str,
) -> Result<()> {
    let (map_tx, map_rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
        let _ = map_tx.send(res);
    });

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    queue.on_submitted_work_done(move || {
        let _ = done_tx.send(());
    });

    let deadline = tokio::time::Instant::now() + GPU_MAP_TIMEOUT;
    let mut map_ok = false;
    let mut gpu_done = false;

    loop {
        device.poll(wgpu::Maintain::Poll);

        if !map_ok {
            if let Ok(result) = map_rx.try_recv() {
                if result.is_err() {
                    return Err(Error::BufferMapFailed);
                }
                map_ok = true;
            }
        }

        if !gpu_done && done_rx.try_recv().is_ok() {
            gpu_done = true;
        }

        if map_ok && gpu_done {
            return Ok(());
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(Error::GpuDeviceError {
                reason: format!(
                    "GPU buffer map timed out after {}s: {timeout_msg}",
                    GPU_MAP_TIMEOUT.as_secs()
                ),
            });
        }
        tokio::task::yield_now().await;
    }
}

pub(crate) async fn read_matches(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    count_staging: &wgpu::Buffer,
    match_staging: &wgpu::Buffer,
    pattern_ids: Option<&[usize]>,
    base_offset: usize,
    max_matches: u32,
    input_len: usize,
) -> Result<Vec<Match>> {
    await_buffer_map(
        device,
        queue,
        count_staging,
        "waiting for GPU match count buffer",
    )
    .await?;

    let count_data = count_staging.slice(..).get_mapped_range();
    let count_array: &[u32] = bytemuck::cast_slice(&count_data);
    if count_array.len() < 2 {
        drop(count_data);
        count_staging.unmap();
        return Err(Error::GpuDeviceError {
            reason: "GPU count buffer too small (expected >= 8 bytes). Fix: verify GPU shader writes match_count and overflow_flag.".to_string(),
        });
    }
    let match_count = count_array[0] as usize;
    let overflow_flag = count_array[1];
    drop(count_data);
    count_staging.unmap();

    if overflow_flag != 0 {
        return Err(Error::MatchBufferOverflow {
            count: match_count,
            max: max_matches as usize,
        });
    }
    if match_count == 0 {
        return Ok(Vec::new());
    }

    await_buffer_map(
        device,
        queue,
        match_staging,
        "waiting for GPU match buffer",
    )
    .await?;

    let match_data = match_staging.slice(..).get_mapped_range();
    let raw: &[u32] = bytemuck::cast_slice(&match_data);

    if match_count
        .checked_mul(4)
        .map_or(true, |needed| needed > raw.len())
    {
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
        let gpu_start = raw[base + 1] as usize;
        let gpu_end = raw[base + 2] as usize;

        // Validate match offsets against the chunk input length — corrupted GPU output
        // could return offsets past the end of the input buffer.
        if gpu_end > input_len {
            tracing::warn!(
                gpu_start,
                gpu_end,
                input_len,
                "GPU returned match offset past input length, skipping"
            );
            continue;
        }

        let start_offset = base_offset.checked_add(gpu_start).ok_or(Error::InputTooLarge {
            bytes: usize::MAX,
            max_bytes: u32::MAX as usize,
        })?;
        let end_offset = base_offset.checked_add(gpu_end).ok_or(Error::InputTooLarge {
            bytes: usize::MAX,
            max_bytes: u32::MAX as usize,
        })?;
        let start = u32::try_from(start_offset).map_err(|_| Error::InputTooLarge {
            bytes: start_offset,
            max_bytes: u32::MAX as usize,
        })?;
        let end = u32::try_from(end_offset).map_err(|_| Error::InputTooLarge {
            bytes: end_offset,
            max_bytes: u32::MAX as usize,
        })?;

        let user_pattern_id = match pattern_ids {
            Some(ids) => {
                let gpu_pattern_idx = raw[base] as usize;
                let Some(&id) = ids.get(gpu_pattern_idx) else {
                    return Err(Error::GpuDeviceError {
                        reason: format!(
                            "GPU returned out-of-bounds pattern index {} (pattern_count={}). Fix: GPU/CPU state mismatch indicates a critical bug.",
                            gpu_pattern_idx,
                            ids.len()
                        ),
                    });
                };
                id as u32
            }
            None => raw[base],
        };

        matches.push(Match {
            pattern_id: user_pattern_id,
            start,
            end,
            padding: 0,
        });
    }
    drop(match_data);
    match_staging.unmap();
    Ok(matches)
}
