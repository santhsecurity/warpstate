use crate::config::DEFAULT_MAX_MATCHES;
use crate::error::{Error, Result};
use crate::rolling_hash::HashPipelineState;
use crate::{GpuMatch, Match};

pub fn readback_results(device: &wgpu::Device, state: &HashPipelineState) -> Result<Vec<Match>> {
    let (tx_count, rx_count) = std::sync::mpsc::channel();
    state.count_staging.slice(..).map_async(
        wgpu::MapMode::Read,
        move |result: std::result::Result<(), wgpu::BufferAsyncError>| {
            let _ = tx_count.send(result);
        },
    );
    wait_for_mapping(device, &rx_count)?;

    let count_data = state.count_staging.slice(..).get_mapped_range();
    let counts: &[u32] = bytemuck::cast_slice(&count_data);
    if counts.len() < 2 {
        drop(count_data);
        state.count_staging.unmap();
        return Err(Error::GpuDeviceError {
            reason: "rolling-hash count buffer too small".to_string(),
        });
    }
    let match_count = counts[0] as usize;
    let overflow = counts[1];
    drop(count_data);
    state.count_staging.unmap();

    if overflow != 0 {
        return Err(Error::MatchBufferOverflow {
            count: match_count,
            max: DEFAULT_MAX_MATCHES as usize,
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
    wait_for_mapping(device, &rx_match)?;

    let match_data = state.match_staging.slice(..).get_mapped_range();
    let raw: &[GpuMatch] = bytemuck::cast_slice(&match_data);
    let valid_count = match_count.min(DEFAULT_MAX_MATCHES as usize);
    let matches = raw[..valid_count]
        .iter()
        .copied()
        .map(Match::from)
        .collect();
    drop(match_data);
    state.match_staging.unmap();

    Ok(matches)
}

fn wait_for_mapping(
    device: &wgpu::Device,
    receiver: &std::sync::mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
) -> Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        device.poll(wgpu::Maintain::Poll);
        match receiver.try_recv() {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(Error::BufferMapFailed);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if std::time::Instant::now() >= deadline {
                    return Err(Error::GpuDeviceError {
                        reason: "GPU buffer map timed out after 30s".to_string(),
                    });
                }
                std::thread::yield_now();
            }
        }
    }
}
