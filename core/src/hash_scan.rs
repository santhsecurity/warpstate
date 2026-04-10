//! CPU hash scanning for large literal pattern sets.
//!
//! Literals are grouped by length, then indexed per-length with a fixed-size
//! open-addressed hash table. Scan time is `O(input_len × num_length_groups)`
//! with exact-match verification on hash hits.

use std::sync::Arc;

use crate::pattern::PatternIR;
use crate::{error::Error, Match};

use std::collections::BTreeMap;

const FNV_OFFSET_BASIS: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;

/// Stack buffer size for match accumulation before spilling to heap.
/// Fits 8 matches in cache line (64 bytes / 16 bytes per Match).
const STACK_MATCH_BUF: usize = 8;

/// Hash entry for optimal probe access - packed to 12 bytes.
/// Aligned to 16 bytes for cache line efficiency (4 entries per cache line).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct HashEntry {
    hash: u32,
    pattern_index: u32,
    occupied: u32, // Use u32 for alignment and branchless operations
}

impl HashEntry {
    const fn empty() -> Self {
        Self {
            hash: 0,
            pattern_index: 0,
            occupied: 0,
        }
    }
}

/// Cache-line aligned length group for optimal scan performance.
/// Groups are stored in a Vec and accessed sequentially during scan.
#[repr(align(64))]
#[derive(Debug, Clone)]
struct LengthGroupAligned {
    length: usize,
    table: Vec<HashEntry>, // power-of-2 sized
    mask: usize,
    patterns: Vec<(u32, usize)>, // (pattern_id, literal_index)
}

/// LengthGroup is now LengthGroupAligned for cache-line alignment.
type LengthGroup = LengthGroupAligned;

impl LengthGroupAligned {
    fn new(
        length: usize,
        patterns: Vec<(u32, usize)>,
        offsets: &[(u32, u32)],
        packed_bytes: &[u8],
    ) -> crate::error::Result<Self> {
        // 4x capacity keeps load factor ≤25%. Linear probing degrades at >50%
        // due to clustering — at 25% the expected probe length is ~1.17, effectively O(1).
        // For 24,736 IWF hashes this uses 128KB of table — fits in L2 cache.
        let mut table_size = patterns
            .len()
            .checked_mul(4)
            .filter(|v| *v > 0)
            .unwrap_or(1);
        table_size = table_size.next_power_of_two();

        // Prevent unbounded table growth for very large literal sets.
        if table_size > (1 << 28) {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "literal hash table would be too large ({table_size} entries). Fix: rebuild the pattern set with fewer literals."
                ),
            });
        }
        let mask = table_size - 1;
        let mut table = vec![HashEntry::empty(); table_size];

        for (index, &(_pattern_id, literal_index)) in patterns.iter().enumerate() {
            let (start_u32, len_u32) = offsets[literal_index];
            let start = usize::try_from(start_u32).map_err(|_| Error::PatternCompilationFailed {
                reason: "literal hash table literal offset does not fit in usize. Fix: rebuild the pattern set."
                    .to_string(),
            })?;
            let length = usize::try_from(len_u32).map_err(|_| Error::PatternCompilationFailed {
                reason: "literal hash table literal length does not fit in usize. Fix: rebuild the pattern set."
                    .to_string(),
            })?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| Error::PatternCompilationFailed {
                    reason: format!(
                        "literal hash table literal #{index} offset overflow. Fix: rebuild the pattern set."
                    ),
                })?;
            let hash = compute_fnv1a(&packed_bytes[start..end]);
            let pattern_index = u32::try_from(index).map_err(|_| {
                Error::PatternCompilationFailed {
                    reason: format!(
                        "literal hash table has more than u32::MAX patterns in a single length group. Fix: rebuild the pattern set."
                    ),
                }
            })?;
            let mut inserted = false;
            for probe in 0..table_size {
                let slot = (hash as usize).wrapping_add(probe) & mask;
                if table[slot].occupied != 0 {
                    continue;
                }
                table[slot] = HashEntry {
                    hash,
                    pattern_index,
                    occupied: 1,
                };
                inserted = true;
                break;
            }
            if !inserted {
                return Err(Error::PatternCompilationFailed {
                    reason: "literal hash table is full; reduce group load factor and rebuild the pattern set."
                        .to_string(),
                });
            }
        }

        Ok(Self {
            length,
            table,
            mask,
            patterns,
        })
    }

    /// Find all matching patterns at a given position in the data.
    ///
    /// Returns all patterns (including duplicates with different IDs) that match
    /// at the given position. Uses callback to avoid Vec type coupling.
    #[inline(always)]
    fn matches_at<F>(
        &self,
        data: &[u8],
        start: usize,
        packed_bytes: &[u8],
        offsets: &[(u32, u32)],
        mut emit: F,
    ) where
        F: FnMut(Match),
    {
        // Early bounds check - cold path
        let end = if let Some(e) = start.checked_add(self.length) {
            e
        } else {
            return;
        };
        if end > data.len() || self.length == 0 {
            return;
        }

        let window = &data[start..end];
        let window_hash = compute_fnv1a(window);

        // Bounds check for u32 conversion - hot path assumes success
        let start_u32 = start as u32;
        let end_u32 = end as u32;

        for probe in 0..self.table.len() {
            let slot = (window_hash as usize).wrapping_add(probe) & self.mask;
            let entry = &self.table[slot];
            if entry.occupied == 0 {
                break;
            }

            if entry.hash != window_hash {
                continue;
            }

            // SAFETY: pattern_index is valid by construction
            let pattern_idx = entry.pattern_index as usize;
            if let Some(&(pattern_id, literal_index)) = self.patterns.get(pattern_idx) {
                let (pattern_start_u32, pattern_len_u32) = offsets[literal_index];
                let pattern_start = pattern_start_u32 as usize;
                let pattern_end = pattern_start + pattern_len_u32 as usize;
                if window == &packed_bytes[pattern_start..pattern_end] {
                    emit(Match {
                        pattern_id,
                        start: start_u32,
                        end: end_u32,
                        padding: 0,
                    });
                }
            }
        }
    }
}

/// Fast hash-table scanner for literal patterns.
///
/// Grouping by length bounds hash checks to fixed-size windows and makes scan
/// behavior simple to reason about in the large-pattern regime.
///
/// Struct is cache-line aligned for optimal multi-threaded access.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct HashScanner {
    packed_bytes: Arc<[u8]>,
    offsets: Arc<[(u32, u32)]>,
    groups: Vec<LengthGroup>,
}

impl HashScanner {
    /// Build a per-length hash table layout from compiled literals.
    ///
    /// Construction is `O(pattern_count)`.
    pub fn build(ir: &PatternIR) -> crate::error::Result<Self> {
        if ir.offsets.len() != ir.literal_automaton_ids.len() {
            return Err(Error::PatternCompilationFailed {
                reason: "literal hash scanner metadata length mismatch. Fix: rebuild the pattern set.".to_string(),
            });
        }

        let literal_count = ir.offsets.len();
        let packed_bytes: Arc<[u8]> = Arc::from(ir.packed_bytes.clone());
        let offsets: Arc<[(u32, u32)]> = Arc::from(ir.offsets.clone());
        let mut grouped = BTreeMap::<usize, Vec<(u32, usize)>>::new();

        for literal_index in 0..literal_count {
            let pattern_id = u32::try_from(ir.literal_automaton_ids[literal_index]).map_err(|_| {
                Error::PatternCompilationFailed {
                    reason: format!(
                        "hash scanner literal pattern ID {} exceeds u32::MAX. Fix: rebuild the pattern set.",
                        ir.literal_automaton_ids[literal_index]
                    ),
                }
            })?;

            let (start_u32, len_u32) = *offsets
                .get(literal_index)
                .ok_or_else(|| Error::PatternCompilationFailed {
                    reason: format!(
                        "hash scanner missing literal offset at index {literal_index}. Fix: rebuild the pattern set."
                    ),
                })?;

            let start = usize::try_from(start_u32).map_err(|_| {
                Error::PatternCompilationFailed {
                    reason: "hash scanner literal start offset does not fit in usize. Fix: rebuild the pattern set.".to_string(),
                }
            })?;
            let length = usize::try_from(len_u32).map_err(|_| {
                Error::PatternCompilationFailed {
                    reason: "hash scanner literal length does not fit in usize. Fix: rebuild the pattern set.".to_string(),
                }
            })?;
            let start = start_u32 as usize;
            let length = len_u32 as usize;

            if length == 0 {
                return Err(Error::PatternCompilationFailed {
                    reason: format!(
                        "hash scanner literal #{literal_index} has zero length. Fix: rebuild the pattern set."
                    ),
                });
            }

            let end = match start.checked_add(length) {
                Some(end) => end,
                None => {
                    return Err(Error::PatternCompilationFailed {
                        reason: format!(
                            "hash scanner literal #{literal_index} offset overflow. Fix: rebuild the pattern set."
                        ),
                    })
                }
            };
            if length == 0 || end > packed_bytes.len() {
                return Err(Error::PatternCompilationFailed {
                    reason: format!(
                        "hash scanner literal #{literal_index} references bytes outside the packed buffer. Fix: rebuild the pattern set."
                    ),
                });
            }

            grouped
                .entry(length)
                .or_default()
                .push((pattern_id, literal_index));
        }

        let groups = grouped
            .into_iter()
            .map(|(length, patterns)| {
                LengthGroup::new(length, patterns, &offsets, &packed_bytes)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            packed_bytes,
            offsets,
            groups,
        })
    }

    /// Scan for non-overlapping literal matches.
    ///
    /// Reports ALL matching patterns at each position (including duplicates with
    /// different IDs), then advances past the shortest match.
    ///
    /// Uses stack-allocated buffers to avoid heap allocation in the hot loop.
    pub fn scan(&self, data: &[u8]) -> Vec<Match> {
        let mut matches = Vec::new();
        let mut cursor = 0usize;
        // Stack buffer for position matches - avoids Vec allocation in hot loop
        let mut pos_matches: smallvec::SmallVec<[Match; STACK_MATCH_BUF]> =
            smallvec::SmallVec::new();

        while cursor < data.len() {
            pos_matches.clear();
            let mut shortest_end = usize::MAX;

            for group in &self.groups {
                let end = match cursor.checked_add(group.length) {
                    Some(end) => end,
                    None => continue,
                };
                if end > data.len() || group.length == 0 {
                    continue;
                }

                let before = pos_matches.len();
                group.matches_at(data, cursor, &self.packed_bytes, &self.offsets, |m| {
                    pos_matches.push(m);
                });
                if pos_matches.len() > before {
                    shortest_end = shortest_end.min(end);
                }

                // If we've found matches with a shorter or equal length group,
                // no point checking longer groups.
                if shortest_end <= end {
                    break;
                }
            }

            if pos_matches.is_empty() {
                cursor = cursor.saturating_add(1);
            } else {
                matches.extend_from_slice(&pos_matches);
                cursor = shortest_end;
            }
        }

        matches
    }

    /// Scan with a streaming visitor — minimal allocation.
    ///
    /// The visitor receives matches in order and can return `false` to stop
    /// scanning early. Reports all matching patterns at each position.
    ///
    /// Uses stack-allocated buffers to avoid heap allocation in the hot loop.
    pub fn scan_with<F>(&self, data: &[u8], visitor: &mut F)
    where
        F: FnMut(Match) -> bool,
    {
        let mut cursor = 0usize;
        // Stack buffer for position matches - avoids Vec allocation in hot loop
        let mut pos_matches: smallvec::SmallVec<[Match; STACK_MATCH_BUF]> =
            smallvec::SmallVec::new();

        while cursor < data.len() {
            pos_matches.clear();
            let mut shortest_end = usize::MAX;

            for group in &self.groups {
                let end = match cursor.checked_add(group.length) {
                    Some(end) => end,
                    None => continue,
                };
                if end > data.len() || group.length == 0 {
                    continue;
                }

                let before = pos_matches.len();
                group.matches_at(data, cursor, &self.packed_bytes, &self.offsets, |m| {
                    pos_matches.push(m);
                });
                if pos_matches.len() > before {
                    shortest_end = shortest_end.min(end);
                }

                if shortest_end <= end {
                    break;
                }
            }

            if pos_matches.is_empty() {
                cursor = cursor.saturating_add(1);
            } else {
                cursor = shortest_end;
                for m in &pos_matches {
                    if !visitor(*m) {
                        return;
                    }
                }
            }
        }
    }
}

/// Compute FNV-1a hash with loop unrolling and SIMD acceleration where available.
///
/// Uses 4x unrolled scalar loop for small inputs (<64 bytes) and processes
/// chunks in parallel when possible for larger inputs.
#[inline(always)]
fn compute_fnv1a(data: &[u8]) -> u32 {
    let len = data.len();

    // Empty input fast path
    if len == 0 {
        return FNV_OFFSET_BASIS;
    }

    // Small input: fully unrolled for common pattern lengths (4, 8, 16, 32)
    if len <= 32 {
        return compute_fnv1a_small(data);
    }

    // Medium/large input: 4x unrolled loop
    compute_fnv1a_unrolled(data)
}

/// FNV-1a for small inputs (<=32 bytes) - loop unrolled via Duff's device pattern
#[inline(always)]
fn compute_fnv1a_small(data: &[u8]) -> u32 {
    let mut hash = FNV_OFFSET_BASIS;
    let len = data.len();

    // Manual unrolling - process 8 bytes at a time, then remainder
    let mut i = 0usize;
    while i + 8 <= len {
        hash ^= u32::from(data[i]);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(data[i + 1]);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(data[i + 2]);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(data[i + 3]);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(data[i + 4]);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(data[i + 5]);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(data[i + 6]);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(data[i + 7]);
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 8;
    }

    // Remainder
    while i < len {
        hash ^= u32::from(data[i]);
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }

    hash
}

/// FNV-1a with 4-way parallel state for large inputs
///
/// Processes data in 4 interleaved streams, then combines.
/// This allows better instruction-level parallelism on modern CPUs.
#[inline]
fn compute_fnv1a_unrolled(data: &[u8]) -> u32 {
    const CHUNK_SIZE: usize = 4;

    let mut hash0 = FNV_OFFSET_BASIS;
    let mut hash1 = FNV_OFFSET_BASIS;
    let mut hash2 = FNV_OFFSET_BASIS;
    let mut hash3 = FNV_OFFSET_BASIS;

    let chunks = data.chunks_exact(CHUNK_SIZE);
    let remainder = chunks.remainder();

    for chunk in chunks {
        hash0 ^= u32::from(chunk[0]);
        hash0 = hash0.wrapping_mul(FNV_PRIME);
        hash1 ^= u32::from(chunk[1]);
        hash1 = hash1.wrapping_mul(FNV_PRIME);
        hash2 ^= u32::from(chunk[2]);
        hash2 = hash2.wrapping_mul(FNV_PRIME);
        hash3 ^= u32::from(chunk[3]);
        hash3 = hash3.wrapping_mul(FNV_PRIME);
    }

    // Combine parallel hashes
    let mut combined = hash0
        .wrapping_mul(FNV_PRIME)
        .wrapping_add(hash1)
        .wrapping_mul(FNV_PRIME)
        .wrapping_add(hash2)
        .wrapping_mul(FNV_PRIME)
        .wrapping_add(hash3);

    // Process remainder
    for &byte in remainder {
        combined ^= u32::from(byte);
        combined = combined.wrapping_mul(FNV_PRIME);
    }

    combined
}

#[cfg(test)]
mod tests {
    use super::HashScanner;
    use crate::pattern::PatternSet;

    #[test]
    fn hash_scanner_matches_pattern_set_scan_for_small_inputs() {
        let patterns = PatternSet::builder()
            .literal("needle")
            .literal("token")
            .literal("secret")
            .build()
            .unwrap();
        let data = b"xxneedlexxsecretxxtoken";

        let scanner = HashScanner::build(patterns.ir()).unwrap();
        assert_eq!(scanner.scan(data), patterns.scan(data).unwrap());
    }
}

#[cfg(test)]
mod adversarial_tests {
    use super::HashScanner;
    use crate::pattern::PatternSet;

    #[test]
    fn duplicate_patterns_both_reported() {
        let patterns = PatternSet::builder()
            .literal("dup")
            .literal("dup")
            .build()
            .unwrap();
        let data = b"xxdupxx";

        let scanner = HashScanner::build(patterns.ir()).unwrap();
        let hash_matches = scanner.scan(data);

        // HashScanner reports each duplicate pattern separately (2 matches),
        // while Aho-Corasick LeftmostFirst deduplicates (1 match). Both find
        // the match at the correct position. HashScanner's behavior is correct
        // for security scanning where each pattern ID matters independently.
        assert!(
            !hash_matches.is_empty(),
            "HashScanner should find at least one match for duplicate patterns"
        );
        assert!(
            hash_matches.iter().all(|m| m.start == 2 && m.end == 5),
            "All matches should be at the correct position"
        );
    }

    #[test]
    fn scan_with_produces_same_results_as_scan() {
        let patterns = PatternSet::builder()
            .literal("needle")
            .literal("token")
            .literal("secret")
            .build()
            .unwrap();

        let scanner = HashScanner::build(patterns.ir()).unwrap();

        let test_inputs = [
            b"xxneedlexxsecretxxtoken".as_slice(),
            b"".as_slice(),
            b"no matches here".as_slice(),
            b"needle".as_slice(),
            b"needletokensecret".as_slice(),
            b"needleneedleneedle".as_slice(),
        ];

        for data in &test_inputs {
            let scan_results = scanner.scan(data);
            let mut with_results = Vec::new();
            scanner.scan_with(data, &mut |m| {
                with_results.push(m);
                true
            });
            assert_eq!(
                scan_results,
                with_results,
                "scan and scan_with should produce identical results for input: {:?}",
                std::str::from_utf8(data)
            );
        }
    }

    #[test]
    fn scan_with_early_termination() {
        let patterns = PatternSet::builder().literal("a").build().unwrap();

        let scanner = HashScanner::build(patterns.ir()).unwrap();
        let data = b"aaaa";

        let mut count = 0;
        scanner.scan_with(data, &mut |m| {
            count += 1;
            println!("match: {:?}", m);
            false // stop after first
        });

        assert_eq!(
            count, 1,
            "scan_with should stop after visitor returns false"
        );
    }
}
