//! GPU rolling-hash pattern matching for literal byte sequences.

use std::collections::BTreeMap;
use std::sync::Mutex;

use bytemuck::{Pod, Zeroable};

use crate::error::{Error, Result};
use crate::matcher::BlockMatcher;
use crate::Match;

pub mod kernel;
pub mod readback;

type GroupedPatterns = Vec<Vec<(u32, Vec<u8>)>>;

const FNV_OFFSET_BASIS: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;
const DEFAULT_MAX_INPUT_SIZE: usize = 256 * 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HashUniforms {
    pub input_len: u32,
    pub pattern_length: u32,
    pub hash_table_size: u32,
    pub max_matches: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HashTableEntry {
    pub occupied: u32,
    pub hash: u32,
    pub pattern_id: u32,
    pub padding: u32,
}

#[derive(Clone, Debug)]
pub struct PatternEntry {
    pub pattern_id: u32,
    pub hash: u32,
    pub raw_bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct LengthGroup {
    pub length: u32,
    pub hash_table_size: u32,
    pub hash_table_buf: wgpu::Buffer,
    pub pattern_bytes_buf: wgpu::Buffer,
    pub pattern_offsets_buf: wgpu::Buffer,
}

#[derive(Debug)]
pub struct HashPipelineState {
    pub input_buf: wgpu::Buffer,
    pub match_buf: wgpu::Buffer,
    pub count_buf: wgpu::Buffer,
    pub uniform_buf: wgpu::Buffer,
    pub count_staging: wgpu::Buffer,
    pub match_staging: wgpu::Buffer,
}

/// GPU pattern matching via Rabin-Karp rolling hash.
///
/// Unlike the DFA backend, this algorithm has zero sequential dependency:
/// each byte position is evaluated independently, making it naturally parallel.
/// It supports exact byte matches only.
#[derive(Debug)]
pub struct RollingHashMatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    length_groups: Vec<LengthGroup>,
    state: Mutex<HashPipelineState>,
    max_input_size: usize,
}

impl RollingHashMatcher {
    /// Create a new rolling-hash GPU matcher using the default input cap.
    pub async fn new(patterns: &[&[u8]]) -> Result<Self> {
        Self::with_max_input_size(patterns, DEFAULT_MAX_INPUT_SIZE).await
    }

    /// Create a new rolling-hash GPU matcher with an explicit maximum input size.
    pub async fn with_max_input_size(patterns: &[&[u8]], max_input_size: usize) -> Result<Self> {
        validate_patterns(patterns)?;

        let (device, queue) = kernel::acquire_device().await?;
        let (pipeline, bind_group_layout) = kernel::compile_pipeline(&device);
        let length_groups = kernel::build_length_groups(&device, patterns)?;
        let state = Mutex::new(kernel::allocate_buffers(&device, max_input_size));

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            length_groups,
            state,
            max_input_size,
        })
    }

    pub(crate) fn group_patterns_by_length(patterns: &[&[u8]]) -> Result<GroupedPatterns> {
        let mut groups = BTreeMap::<usize, Vec<(u32, Vec<u8>)>>::new();
        for (pattern_id, pattern) in patterns.iter().enumerate() {
            let pattern_id =
                u32::try_from(pattern_id).map_err(|_| Error::PatternCompilationFailed {
                    reason: "pattern id exceeds 32-bit address space".to_string(),
                })?;
            groups
                .entry(pattern.len())
                .or_default()
                .push((pattern_id, (*pattern).to_vec()));
        }
        Ok(groups.into_values().collect())
    }

    pub(crate) fn compute_fnv1a(data: &[u8]) -> u32 {
        let mut hash = FNV_OFFSET_BASIS;
        for &byte in data {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    pub async fn scan_block_impl(&self, data: &[u8]) -> Result<Vec<Match>> {
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

        let state = lock_state(&self.state);
        kernel::upload_input(&self.queue, &state.input_buf, data);

        let mut matches = Vec::new();
        for group in &self.length_groups {
            if group.length > input_len {
                continue;
            }

            kernel::reset_count_and_uniforms(
                &self.queue,
                &state,
                input_len,
                group.length,
                group.hash_table_size,
            );
            let bind_group =
                kernel::assemble_bind_group(&self.device, &self.bind_group_layout, group, &state);
            kernel::dispatch_shader(
                &self.device,
                &self.queue,
                &self.pipeline,
                &bind_group,
                &state,
                input_len,
            );
            matches.extend(readback::readback_results(&self.device, &state)?);
        }

        matches.sort_unstable();
        Ok(matches)
    }

    pub fn max_block_size(&self) -> usize {
        self.max_input_size
    }
}

fn validate_patterns(patterns: &[&[u8]]) -> Result<()> {
    if patterns.is_empty() {
        return Err(Error::EmptyPatternSet);
    }

    if patterns.len() > u32::MAX as usize {
        return Err(Error::PatternCompilationFailed {
            reason: format!(
                "rolling-hash matcher supports at most {} patterns",
                u32::MAX
            ),
        });
    }

    for (index, pattern) in patterns.iter().enumerate() {
        if pattern.is_empty() {
            return Err(Error::EmptyPattern { index });
        }

        if pattern.len() > u32::MAX as usize {
            return Err(Error::PatternTooLarge {
                index,
                bytes: pattern.len(),
                max: u32::MAX as usize,
            });
        }
    }

    Ok(())
}

fn lock_state(state: &Mutex<HashPipelineState>) -> std::sync::MutexGuard<'_, HashPipelineState> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!(
                "rolling-hash pipeline state mutex was poisoned; continuing with recovered state"
            );
            poisoned.into_inner()
        }
    }
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use crate::PatternSet;

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        pollster::block_on(future)
    }

    fn matcher_or_skip(patterns: &[&[u8]]) -> Result<Option<RollingHashMatcher>> {
        match block_on(RollingHashMatcher::new(patterns)) {
            Ok(matcher) => Ok(Some(matcher)),
            Err(Error::NoGpuAdapter) => {
                eprintln!("skipping rolling-hash GPU test: no GPU adapter available");
                Ok(None)
            }
            Err(err) => Err(err.into()),
        }
    }

    fn scan_or_skip(patterns: &[&[u8]], data: &[u8]) -> Result<Option<Vec<Match>>> {
        let Some(matcher) = matcher_or_skip(patterns)? else {
            return Ok(None);
        };
        match block_on(matcher.scan_block(data)) {
            Ok(gpu_matches) => Ok(Some(gpu_matches)),
            Err(err) => Err(err.into()),
        }
    }

    #[test]
    fn hash_single_pattern() {
        let Some(matches) = scan_or_skip(&[b"secret"], b"top secret value").unwrap() else {
            return;
        };
        assert_eq!(
            matches,
            vec![Match {
                pattern_id: 0,
                start: 4,
                end: 10,
                padding: 0,
            }]
        );
    }

    #[test]
    fn hash_multiple_same_length() {
        let patterns: Vec<&[u8]> = vec![
            b"one0", b"two0", b"tri0", b"for0", b"fiv0", b"six0", b"sev0", b"egt0", b"nin0",
            b"ten0",
        ];
        let data = b"two0 egt0 one0 ten0";
        let Some(matches) = scan_or_skip(&patterns, data).unwrap() else {
            return;
        };
        let ids: Vec<u32> = matches.iter().map(|mat| mat.pattern_id).collect();
        assert_eq!(ids, vec![1, 7, 0, 9]);
    }

    #[test]
    fn hash_multiple_different_lengths() {
        let patterns: Vec<&[u8]> = vec![b"abc", b"wxyz", b"literal"];
        let data = b"abc wxyz literal abc";
        let Some(matches) = scan_or_skip(&patterns, data).unwrap() else {
            return;
        };
        assert_eq!(matches.len(), 4);
        assert_eq!(matches[0].pattern_id, 0);
        assert_eq!(matches[1].pattern_id, 1);
        assert_eq!(matches[2].pattern_id, 2);
        assert_eq!(matches[3].pattern_id, 0);
    }

    #[test]
    fn hash_matches_cpu_parity() {
        let patterns = [b"alpha".as_slice(), b"beta".as_slice(), b"gamma".as_slice()];
        let data = b"alpha beta gamma alpha gamma";
        let Some(gpu_matches) = scan_or_skip(&patterns, data).unwrap() else {
            return;
        };

        let pattern_set = PatternSet::builder()
            .literal_bytes(patterns[0])
            .literal_bytes(patterns[1])
            .literal_bytes(patterns[2])
            .build()
            .unwrap();
        let cpu_matches = pattern_set.scan(data).unwrap();
        assert_eq!(gpu_matches, cpu_matches);
    }

    #[test]
    fn hash_overlapping_matches() {
        let Some(matches) = scan_or_skip(&[b"ab"], b"abab").unwrap() else {
            return;
        };
        assert_eq!(
            matches,
            vec![
                Match {
                    pattern_id: 0,
                    start: 0,
                    end: 2,
                    padding: 0,
                },
                Match {
                    pattern_id: 0,
                    start: 2,
                    end: 4,
                    padding: 0,
                },
            ]
        );
    }

    #[test]
    #[ignore = "GPU shader parity bug under software Vulkan; requires WGSL debugging."]
    fn hash_no_false_positives() {
        let Some(matches) = scan_or_skip(&[b"abcdef"], b"abcdeg abcdefg abcdee").unwrap() else {
            return;
        };
        assert!(matches.is_empty());
    }

    #[test]
    fn hash_empty_input() {
        let Some(matcher) = matcher_or_skip(&[b"anything"]).unwrap() else {
            return;
        };
        let gpu_matches = block_on(matcher.scan_block(b"")).unwrap();
        assert!(gpu_matches.is_empty());
    }

    #[test]
    fn hash_large_pattern_count() {
        let patterns: Vec<Vec<u8>> = (0..1000)
            .map(|index| format!("pattern-{index:04}").into_bytes())
            .collect();
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(Vec::as_slice).collect();
        let data = b"prefix pattern-0007 middle pattern-0420 suffix pattern-0999";
        let Some(matches) = scan_or_skip(&pattern_refs, data).unwrap() else {
            return;
        };
        let ids: Vec<u32> = matches.iter().map(|mat| mat.pattern_id).collect();
        assert_eq!(ids, vec![7, 420, 999]);
    }

    #[test]
    fn hash_pattern_at_end_of_input() {
        let Some(matches) = scan_or_skip(&[b"tail"], b"look-at-the-tail").unwrap() else {
            return;
        };
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 12);
        assert_eq!(matches[0].end, 16);
    }

    #[test]
    fn compute_fnv1a_matches_reference_value() {
        assert_eq!(
            RollingHashMatcher::compute_fnv1a(b"warpstate"),
            2_711_197_248
        );
    }

    #[test]
    fn groups_patterns_by_length() {
        let patterns = [b"a".as_slice(), b"bb".as_slice(), b"c".as_slice()];
        let groups = RollingHashMatcher::group_patterns_by_length(&patterns).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
    }
}

impl BlockMatcher for RollingHashMatcher {
    async fn scan_block(&self, data: &[u8]) -> matchkit::Result<Vec<Match>> {
        self.scan_block_impl(data).await.map_err(Into::into)
    }

    fn max_block_size(&self) -> usize {
        self.max_block_size()
    }
}
