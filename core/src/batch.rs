//! Batch scan API for processing multiple data items in a single call.
//!
//! The batch API is the foundation for GPU coalescing: many small inputs
//! are scanned together, and matches are attributed back to their source via
//! caller-provided identifiers.

use crate::error::Result;
use crate::Match;
use std::cmp::Reverse;

/// A single item in a batch scan request.
#[derive(Debug, Clone, Copy)]
pub struct ScanItem<'a> {
    /// Caller-chosen identifier to attribute matches back to their source.
    pub id: u64,
    /// The data to scan.
    pub data: &'a [u8],
}

/// A match result tagged with the source item it originated from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaggedMatch {
    /// The `id` from the [`ScanItem`] that produced this match.
    pub source_id: u64,
    /// The pattern match details. Offsets are relative to the source item.
    pub matched: Match,
}

/// Batch scan configuration.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of inputs per dispatch group.
    pub max_inputs: usize,
    /// Maximum total bytes per dispatch group.
    pub max_bytes: usize,
    /// Whether to scan larger inputs first for better utilization.
    pub sort_by_size: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_inputs: 4096,
            max_bytes: 256 * 1024 * 1024,
            sort_by_size: true,
        }
    }
}

/// Coalescing metadata for mapping GPU buffer offsets back to source items.
#[derive(Debug)]
pub struct CoalesceMap {
    /// `(source_id, start_offset_in_buffer, length)`.
    pub entries: Vec<(u64, usize, usize)>,
    /// The coalesced buffer ready for GPU dispatch.
    pub buffer: Vec<u8>,
}

/// Coalesce multiple scan items into a single contiguous buffer for GPU dispatch.
///
/// # Errors
///
/// Returns [`Error::InputTooLarge`] if the total coalesced size exceeds `u32::MAX`,
/// which is the GPU backend's addressing limit.
pub fn coalesce(items: &[ScanItem<'_>]) -> crate::error::Result<CoalesceMap> {
    let mut total_size: usize = 0;
    for item in items {
        total_size =
            total_size
                .checked_add(item.data.len())
                .ok_or(crate::error::Error::InputTooLarge {
                    bytes: usize::MAX,
                    max_bytes: u32::MAX as usize,
                })?;
    }
    if total_size > u32::MAX as usize {
        return Err(crate::error::Error::InputTooLarge {
            bytes: total_size,
            max_bytes: u32::MAX as usize,
        });
    }
    let mut buffer = Vec::with_capacity(total_size);
    let mut entries = Vec::with_capacity(items.len());

    for item in items {
        let start = buffer.len();
        buffer.extend_from_slice(item.data);
        entries.push((item.id, start, item.data.len()));
    }

    Ok(CoalesceMap { entries, buffer })
}

/// Map matches from a coalesced buffer back to their source items.
///
/// Uses a cursor-based linear scan instead of binary search per match.
/// Matches arrive in sorted order (same order as the coalesced buffer),
/// so sequential matches almost always belong to the same or next item.
/// This is O(matches + items) instead of O(matches × log(items)).
pub fn decoalesce(map: &CoalesceMap, global_matches: Vec<Match>) -> Vec<TaggedMatch> {
    let mut tagged = Vec::with_capacity(global_matches.len());
    let mut cursor = 0usize;

    for m in global_matches {
        let global_start = m.start as usize;
        let global_end = m.end as usize;

        // Advance cursor to the item containing this match.
        // Matches are sorted by position, so cursor only moves forward.
        while cursor < map.entries.len() {
            let (_, offset, len) = map.entries[cursor];
            if global_start < offset + len {
                break;
            }
            cursor += 1;
        }
        if cursor >= map.entries.len() {
            break;
        }

        let (_, item_offset, _item_len) = map.entries[cursor];
        if global_start < item_offset {
            continue; // match is before current item (shouldn't happen with sorted matches)
        }
        let entry_idx = cursor;

        let (source_id, item_offset, item_len) = map.entries[entry_idx];

        // Reject matches that cross item boundaries — these are GPU coalescing
        // artifacts, not real matches within the source item.
        if global_end > item_offset + item_len {
            continue;
        }

        let local_start = global_start - item_offset;
        let local_end = global_end - item_offset;
        // Skip matches that can't fit in u32 (items >4GB).
        let Ok(start) = u32::try_from(local_start) else {
            continue;
        };
        let Ok(end) = u32::try_from(local_end) else {
            continue;
        };
        tagged.push(TaggedMatch {
            source_id,
            matched: Match {
                pattern_id: m.pattern_id,
                start,
                end,
                padding: 0,
            },
        });
    }

    tagged.sort_unstable_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then(a.matched.start.cmp(&b.matched.start))
            .then(a.matched.pattern_id.cmp(&b.matched.pattern_id))
            .then(a.matched.end.cmp(&b.matched.end))
    });
    tagged
}

/// CPU batch scan: scans each item independently using the pattern set.
pub fn scan_batch_cpu<'a>(
    patterns: &crate::PatternSet,
    items: impl IntoIterator<Item = ScanItem<'a>>,
) -> Result<Vec<TaggedMatch>> {
    let mut tagged = Vec::new();

    for item in items {
        let matches = patterns.scan(item.data)?;
        for m in matches {
            tagged.push(TaggedMatch {
                source_id: item.id,
                matched: m,
            });
        }
    }

    tagged.sort_unstable_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then(a.matched.start.cmp(&b.matched.start))
            .then(a.matched.pattern_id.cmp(&b.matched.pattern_id))
            .then(a.matched.end.cmp(&b.matched.end))
    });

    Ok(tagged)
}

fn planned_batches(inputs: &[&[u8]], config: &BatchConfig) -> Vec<Vec<usize>> {
    if inputs.is_empty() {
        return Vec::new();
    }

    let max_inputs = config.max_inputs.max(1);
    let max_bytes = config.max_bytes.max(1);
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    if config.sort_by_size {
        order.sort_unstable_by_key(|&idx| Reverse(inputs[idx].len()));
    }

    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0usize;

    for index in order {
        let len = inputs[index].len();
        let would_exceed = !current.is_empty()
            && (current.len() >= max_inputs || current_bytes.saturating_add(len) > max_bytes);
        if would_exceed {
            batches.push(std::mem::take(&mut current));
            current_bytes = 0;
        }

        current.push(index);
        current_bytes = current_bytes.saturating_add(len);

        if current.len() >= max_inputs || current_bytes >= max_bytes {
            batches.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
    }

    if !current.is_empty() {
        batches.push(current);
    }

    batches
}

fn tagged_to_nested(tagged: Vec<TaggedMatch>, input_count: usize) -> Vec<Vec<Match>> {
    let mut grouped = vec![Vec::new(); input_count];
    for item in tagged {
        let Ok(index) = usize::try_from(item.source_id) else {
            continue;
        };
        if let Some(bucket) = grouped.get_mut(index) {
            bucket.push(item.matched);
        }
    }
    grouped
}

#[cfg(feature = "gpu")]
async fn scan_nested_with_gpu(
    matcher: &crate::GpuMatcher,
    inputs: &[&[u8]],
    config: &BatchConfig,
) -> Result<Vec<Vec<Match>>> {
    let mut grouped = vec![Vec::new(); inputs.len()];
    for batch in planned_batches(inputs, config) {
        let items: Vec<ScanItem<'_>> = batch
            .iter()
            .map(|&index| ScanItem {
                id: index as u64,
                data: inputs[index],
            })
            .collect();
        let map = coalesce(&items)?;
        if map.buffer.is_empty() {
            continue;
        }
        let global_matches = matcher.scan(&map.buffer).await?;
        let tagged = decoalesce(&map, global_matches);
        let nested = tagged_to_nested(tagged, inputs.len());
        for (index, matches) in nested.into_iter().enumerate() {
            if !matches.is_empty() {
                grouped[index].extend(matches);
            }
        }
    }
    Ok(grouped)
}

async fn scan_nested_with_auto(
    matcher: &crate::AutoMatcher,
    inputs: &[&[u8]],
    config: &BatchConfig,
) -> Result<Vec<Vec<Match>>> {
    let mut grouped = vec![Vec::new(); inputs.len()];
    for batch in planned_batches(inputs, config) {
        let items: Vec<ScanItem<'_>> = batch
            .iter()
            .map(|&index| ScanItem {
                id: index as u64,
                data: inputs[index],
            })
            .collect();
        let map = coalesce(&items)?;
        if map.buffer.is_empty() {
            continue;
        }
        let global_matches = matcher.scan(&map.buffer).await?;
        let tagged = decoalesce(&map, global_matches);
        let nested = tagged_to_nested(tagged, inputs.len());
        for (index, matches) in nested.into_iter().enumerate() {
            if !matches.is_empty() {
                grouped[index].extend(matches);
            }
        }
    }
    Ok(grouped)
}

#[cfg(feature = "gpu")]
/// Submit multiple inputs as a configurable batched GPU scan.
pub async fn scan_batch(
    matcher: &crate::GpuMatcher,
    inputs: &[&[u8]],
    config: &BatchConfig,
) -> Result<Vec<Vec<Match>>> {
    scan_nested_with_gpu(matcher, inputs, config).await
}

/// Submit multiple inputs through the auto router, falling back to CPU when needed.
pub async fn scan_batch_auto(
    matcher: &crate::AutoMatcher,
    inputs: &[&[u8]],
    config: &BatchConfig,
) -> Result<Vec<Vec<Match>>> {
    scan_nested_with_auto(matcher, inputs, config).await
}

/// GPU-coalesced batch scan: merges items into one buffer, dispatches to GPU,
/// then splits matches back to their sources.
///
/// This is how GPU beats CPU on many small files: instead of N separate scans,
/// one coalesced buffer gets one GPU dispatch at memory bandwidth.
pub async fn scan_batch_gpu<'a>(
    matcher: &crate::AutoMatcher,
    items: impl IntoIterator<Item = ScanItem<'a>>,
) -> Result<Vec<TaggedMatch>> {
    let items: Vec<ScanItem<'_>> = items.into_iter().collect();
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let inputs: Vec<&[u8]> = items.iter().map(|item| item.data).collect();
    let mut grouped = scan_nested_with_auto(matcher, &inputs, &BatchConfig::default()).await?;
    let mut tagged = Vec::new();
    for (index, item) in items.into_iter().enumerate() {
        for matched in grouped[index].drain(..) {
            tagged.push(TaggedMatch {
                source_id: item.id,
                matched,
            });
        }
    }
    tagged.sort_unstable_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then(a.matched.start.cmp(&b.matched.start))
            .then(a.matched.pattern_id.cmp(&b.matched.pattern_id))
            .then(a.matched.end.cmp(&b.matched.end))
    });
    Ok(tagged)
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use crate::PatternSet;

    #[test]
    fn batch_scan_attributes_matches_to_source_items() {
        let patterns = PatternSet::builder().literal("needle").build().unwrap();

        let items = vec![
            ScanItem {
                id: 100,
                data: b"no match here",
            },
            ScanItem {
                id: 200,
                data: b"find the needle",
            },
            ScanItem {
                id: 300,
                data: b"needle at start",
            },
        ];

        let results = scan_batch_cpu(&patterns, items).unwrap();

        // Only items 200 and 300 should have matches.
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].source_id, 200);
        assert_eq!(results[0].matched.start, 9);
        assert_eq!(results[1].source_id, 300);
        assert_eq!(results[1].matched.start, 0);
    }

    #[test]
    fn coalesce_round_trips_correctly() {
        let patterns = PatternSet::builder().literal("abc").build().unwrap();

        let items = vec![
            ScanItem {
                id: 0,
                data: b"xxabcxx",
            },
            ScanItem {
                id: 1,
                data: b"abcabc",
            },
        ];

        let map = coalesce(&items).unwrap();
        let global_matches = patterns.scan(&map.buffer).unwrap();
        let tagged = decoalesce(&map, global_matches);

        // Item 0 has one match at offset 2, item 1 has two matches at 0 and 3.
        assert_eq!(tagged.len(), 3);
        assert_eq!(tagged[0].source_id, 0);
        assert_eq!(tagged[0].matched.start, 2);
        assert_eq!(tagged[1].source_id, 1);
        assert_eq!(tagged[1].matched.start, 0);
        assert_eq!(tagged[2].source_id, 1);
        assert_eq!(tagged[2].matched.start, 3);
    }

    #[test]
    fn decoalesce_rejects_cross_boundary_matches() {
        // A match that spans two items should be rejected.
        let map = CoalesceMap {
            entries: vec![(0, 0, 3), (1, 3, 3)],
            buffer: b"abcdef".to_vec(),
        };

        let cross_boundary = Match {
            pattern_id: 0,
            start: 2,
            end: 5, // spans item 0 (0..3) into item 1 (3..6)
            padding: 0,
        };

        let tagged = decoalesce(&map, vec![cross_boundary]);
        assert!(tagged.is_empty(), "cross-boundary match must be rejected");
    }

    #[test]
    fn empty_batch_produces_no_results() {
        let patterns = PatternSet::builder().literal("test").build().unwrap();

        let results = scan_batch_cpu(&patterns, vec![]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn batch_planner_respects_limits() {
        let inputs = [
            b"aaaa".as_slice(),
            b"bbbbbb".as_slice(),
            b"c".as_slice(),
            b"dddd".as_slice(),
        ];
        let config = BatchConfig {
            max_inputs: 2,
            max_bytes: 7,
            sort_by_size: true,
        };
        assert_eq!(
            planned_batches(&inputs, &config),
            vec![vec![1], vec![0], vec![3, 2]]
        );
    }

    #[test]
    fn tagged_matches_group_back_into_input_order() {
        let grouped = tagged_to_nested(
            vec![
                TaggedMatch {
                    source_id: 2,
                    matched: Match {
                        pattern_id: 7,
                        start: 1,
                        end: 4,
                        padding: 0,
                    },
                },
                TaggedMatch {
                    source_id: 0,
                    matched: Match {
                        pattern_id: 3,
                        start: 0,
                        end: 2,
                        padding: 0,
                    },
                },
            ],
            3,
        );
        assert_eq!(grouped[0].len(), 1);
        assert_eq!(grouped[1].len(), 0);
        assert_eq!(grouped[2].len(), 1);
        assert_eq!(grouped[0][0].pattern_id, 3);
        assert_eq!(grouped[2][0].pattern_id, 7);
    }
}
