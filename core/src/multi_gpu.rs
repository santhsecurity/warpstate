//! Multi-GPU matcher orchestration over one [`crate::GpuMatcher`] per adapter.

use std::sync::Arc;

use crate::config::AutoMatcherConfig;
use crate::error::{Error, Result};
use crate::gpu::{GpuMatcher, DEFAULT_CHUNK_OVERLAP};
use crate::matcher::Matcher;
use crate::{Match, PatternSet};

/// Scans inputs across every available GPU adapter discovered by `wgpu`.
///
/// `MultiGpuMatcher` builds one [`GpuMatcher`] per usable non-CPU adapter and
/// shards each scan into suffix-overlapped byte ranges. Each device scans its
/// assigned shard in parallel, then the matcher restores global offsets and
/// removes duplicate boundary matches.
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example() -> warpstate::Result<()> {
/// use warpstate::{MultiGpuMatcher, PatternSet};
///
/// let patterns = PatternSet::builder().literal("needle").build()?;
/// let matcher = MultiGpuMatcher::new(&patterns).await?;
/// let matches = matcher.scan(b"xxneedlezz").await?;
///
/// assert_eq!(matches.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct MultiGpuMatcher {
    devices: Vec<Arc<GpuMatcher>>,
    chunk_overlap: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScanShard {
    device_index: usize,
    start: usize,
    nominal_end: usize,
    scan_end: usize,
}

impl MultiGpuMatcher {
    /// Initialize with all available GPUs. Falls back to single-GPU if only one available.
    pub async fn new(patterns: &PatternSet) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let config = AutoMatcherConfig::default();
        let chunk_overlap = config
            .configured_chunk_overlap()
            .max(
                patterns
                    .ir()
                    .offsets
                    .iter()
                    .map(|&(_, len)| len as usize)
                    .max()
                    .unwrap_or(0),
            )
            .max(DEFAULT_CHUNK_OVERLAP);

        let mut devices = Vec::new();
        for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
            match GpuMatcher::from_adapter(patterns, &adapter, config.clone()).await {
                Ok(Some(matcher)) => devices.push(Arc::new(matcher)),
                Ok(None) => {}
                Err(Error::GpuDeviceError { reason }) => {
                    tracing::warn!(
                        reason,
                        adapter = ?adapter.get_info(),
                        "skipping unusable GPU adapter during multi-GPU initialization"
                    );
                }
                Err(other) => return Err(other),
            }
        }

        if devices.is_empty() {
            return Err(Error::NoGpuAdapter);
        }

        let chunk_overlap = devices
            .iter()
            .map(|device| device.chunk_overlap())
            .max()
            .unwrap_or(chunk_overlap);

        Ok(Self {
            devices,
            chunk_overlap,
        })
    }

    /// Number of GPUs being used.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Scan across all GPUs in parallel from pre-shared data.
    /// Avoids copying when the caller already holds an `Arc<[u8]>`.
    pub async fn scan_shared(&self, input: Arc<[u8]>) -> Result<Vec<Match>> {
        self.scan_inner(input).await
    }

    /// Scan across all GPUs in parallel.
    pub async fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data.len() > u32::MAX as usize {
            return Err(Error::InputTooLarge {
                bytes: data.len(),
                max_bytes: u32::MAX as usize,
            });
        }
        if self.devices.len() == 1 {
            return self.devices[0].scan(data).await;
        }

        let input: Arc<[u8]> = Arc::from(data);
        self.scan_inner(input).await
    }

    async fn scan_inner(&self, input: Arc<[u8]>) -> Result<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        if input.len() > u32::MAX as usize {
            return Err(Error::InputTooLarge {
                bytes: input.len(),
                max_bytes: u32::MAX as usize,
            });
        }
        let shards = build_scan_shards(input.len(), self.devices.len(), self.chunk_overlap);
        if shards.len() <= 1 {
            return self.devices[0].scan(&input).await;
        }

        let mut merged = Vec::new();
        for shard in shards {
            let device = Arc::clone(&self.devices[shard.device_index]);
            let mut shard_matches = device.scan(&input[shard.start..shard.scan_end]).await?;
            let shard_len = shard.nominal_end - shard.start;
            shard_matches.retain(|m| (m.start as usize) < shard_len);
            let shard_offset = u32::try_from(shard.start).map_err(|_| Error::InputTooLarge {
                bytes: shard.start,
                max_bytes: u32::MAX as usize,
            })?;
            for matched in &mut shard_matches {
                matched.start = matched.start.saturating_add(shard_offset);
                matched.end = matched.end.saturating_add(shard_offset);
            }
            merged.extend(shard_matches);
        }

        sort_and_dedup_matches(&mut merged);
        Ok(merged)
    }
}

impl Matcher for MultiGpuMatcher {
    async fn scan(&self, data: &[u8]) -> matchkit::Result<Vec<Match>> {
        Self::scan(self, data).await.map_err(Into::into)
    }
}

fn build_scan_shards(
    input_len: usize,
    device_count: usize,
    chunk_overlap: usize,
) -> Vec<ScanShard> {
    if input_len == 0 || device_count == 0 {
        return Vec::new();
    }

    let active_devices = device_count.min(input_len);
    let mut shards = Vec::with_capacity(active_devices);
    for device_index in 0..active_devices {
        let start = input_len * device_index / active_devices;
        let nominal_end = input_len * (device_index + 1) / active_devices;
        if nominal_end <= start {
            continue;
        }
        let scan_end = if device_index + 1 == active_devices {
            nominal_end
        } else {
            nominal_end.saturating_add(chunk_overlap).min(input_len)
        };
        shards.push(ScanShard {
            device_index,
            start,
            nominal_end,
            scan_end,
        });
    }
    shards
}

fn sort_and_dedup_matches(matches: &mut Vec<Match>) {
    matches.sort_unstable();
    matches.dedup_by(|left, right| {
        left.pattern_id == right.pattern_id && left.start == right.start && left.end == right.end
    });
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;

    #[test]
    fn scan_shards_cover_input_with_boundary_overlap() {
        let shards = build_scan_shards(100, 3, 8);
        assert_eq!(
            shards,
            vec![
                ScanShard {
                    device_index: 0,
                    start: 0,
                    nominal_end: 33,
                    scan_end: 41,
                },
                ScanShard {
                    device_index: 1,
                    start: 33,
                    nominal_end: 66,
                    scan_end: 74,
                },
                ScanShard {
                    device_index: 2,
                    start: 66,
                    nominal_end: 100,
                    scan_end: 100,
                },
            ]
        );
    }

    #[test]
    fn sort_and_dedup_matches_removes_overlap_duplicates() {
        let mut matches = vec![
            Match {
                pattern_id: 1,
                start: 50,
                end: 56,
                padding: 0,
            },
            Match {
                pattern_id: 0,
                start: 10,
                end: 16,
                padding: 0,
            },
            Match {
                pattern_id: 1,
                start: 50,
                end: 56,
                padding: 0,
            },
        ];

        sort_and_dedup_matches(&mut matches);

        assert_eq!(
            matches,
            vec![
                Match {
                    pattern_id: 0,
                    start: 10,
                    end: 16,
                    padding: 0,
                },
                Match {
                    pattern_id: 1,
                    start: 50,
                    end: 56,
                    padding: 0,
                },
            ]
        );
    }

    #[test]
    fn multi_gpu_matcher_matches_cpu_on_available_hardware() {
        let patterns = PatternSet::builder()
            .literal("needle")
            .literal("abc")
            .build()
            .unwrap();
        let matcher = match pollster::block_on(MultiGpuMatcher::new(&patterns)) {
            Ok(matcher) => matcher,
            Err(Error::NoGpuAdapter) => return,
            Err(other) => panic!("multi-GPU init failed unexpectedly: {other:?}"),
        };

        assert!(matcher.device_count() >= 1);

        let input = b"abc-needle-abc-needle";
        let gpu_matches = pollster::block_on(matcher.scan(input)).unwrap();
        let cpu_matches = patterns.scan(input).unwrap();
        assert_eq!(gpu_matches, cpu_matches);
    }
}
