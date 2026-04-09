//! CPU backend for literal and regex pattern matching.
//!
//! Provides two scanning modes:
//! - [`scan`]: Non-overlapping, SIMD-accelerated via Teddy prefilters. Default for CLI.
//! - [`scan_overlapping`]: Reports every match at every byte position. For GPU parity.

use crate::error::{Error, Result};
use crate::hash_scan::HashScanner;
use crate::pattern::PatternIR;
use crate::Match;

pub mod scan;

pub use self::scan::{scan, scan_aho_corasick, scan_count, scan_overlapping, scan_with};

/// Maximum input size the CPU backend supports.
///
/// `Match.start` and `Match.end` are `u32` (required for GPU buffer `#[repr(C)]`
/// alignment parity). Inputs larger than this would silently wrap offsets.
pub(crate) const MAX_CPU_INPUT_BYTES: usize = u32::MAX as usize;
/// Threshold above which we use HashScanner instead of Aho-Corasick.
/// AC is O(N) regardless of pattern count and uses SIMD Teddy prefilters.
/// HashScanner is O(N × length_groups) — slower but uses less memory.
/// At 100K patterns AC uses ~100-200MB which is acceptable for a scanner
/// that runs on machines with 16GB+ RAM. Only fall back to HashScanner
/// for truly massive pattern sets (>500K) where AC memory becomes a concern.
// Threshold for using HashScanner instead of Aho-Corasick.
// Benchmarks show crossover point is around 5000-10000 patterns:
// - 1000 patterns: Aho-Corasick ~180 MB/s, HashScanner ~110 MB/s (AC wins)
// - 10000 patterns: Aho-Corasick ~104 MB/s, HashScanner ~120 MB/s (Hash wins)
// HashScanner has higher overhead but better scaling for very large pattern sets.
pub(crate) const HASH_SCANNER_LITERAL_THRESHOLD: usize = 5_000;

/// Maximum matches before the CPU scan returns an overflow error.
/// Matches the GPU match buffer limit for backend parity.
pub(crate) const MAX_CPU_MATCHES: usize = 1_048_576;

pub(crate) fn estimated_match_capacity(data_len: usize) -> usize {
    data_len.clamp(64, 1_000_000)
}

pub(crate) fn check_input_size(data: &[u8]) -> Result<()> {
    if data.len() > MAX_CPU_INPUT_BYTES {
        return Err(Error::InputTooLarge {
            bytes: data.len(),
            max_bytes: MAX_CPU_INPUT_BYTES,
        });
    }
    Ok(())
}

fn match_sort_key(item: &Match) -> (u32, u32, u32) {
    // Match fields are already u32, these casts are no-ops but kept for type consistency
    (item.start, item.pattern_id, item.end)
}

fn matches_are_sorted(matches: &[Match]) -> bool {
    matches
        .windows(2)
        .all(|window| match_sort_key(&window[0]) <= match_sort_key(&window[1]))
}

pub(crate) fn sort_matches_if_needed(matches: &mut [Match]) {
    if matches_are_sorted(matches) {
        return;
    }

    matches.sort_unstable_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then(left.pattern_id.cmp(&right.pattern_id))
            .then(left.end.cmp(&right.end))
    });
}

pub(crate) fn finish_matches(mut matches: Vec<Match>) -> Result<Vec<Match>> {
    let count = matches.len();
    if count > MAX_CPU_MATCHES {
        matches.truncate(MAX_CPU_MATCHES);
        return Err(Error::MatchBufferOverflow {
            count,
            max: MAX_CPU_MATCHES,
        });
    }
    Ok(matches)
}

/// Reusable CPU scanner for repeated scans with the same compiled pattern IR.
///
/// When the IR is a single case-sensitive literal with no regex DFAs, this
/// caches a pre-built `memchr::memmem::Finder` so repeated scans avoid
/// rebuilding the search tables on every call.
///
/// # Examples
///
/// ```rust
/// use warpstate::{CachedScanner, Match, PatternSet};
///
/// let patterns = PatternSet::builder().literal("needle").build().unwrap();
/// let scanner = CachedScanner::new(patterns.ir()).unwrap();
/// let mut out = [Match::from_parts(0, 0, 0); 16];
/// let count = scanner.scan(b"needle in a haystack", &mut out).unwrap();
/// assert_eq!(count, 1);
/// ```
pub struct CachedScanner<'a> {
    inner: CachedScannerInner<'a>,
}

#[allow(clippy::large_enum_variant)]
enum CachedScannerInner<'a> {
    SingleLiteral {
        finder: memchr::memmem::Finder<'a>,
        pattern_id: u32,
        needle_len: u32,
    },
    Hash {
        scanner: HashScanner,
    },
    Generic {
        ir: &'a PatternIR,
    },
}

impl<'a> CachedScanner<'a> {
    /// Build a reusable scanner for the provided compiled pattern IR.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PatternCompilationFailed`] if the pattern metadata is
    /// internally inconsistent.
    pub fn new(ir: &'a PatternIR) -> Result<Self> {
        if ir.offsets.len() == 1 && ir.regex_dfas.is_empty() && !ir.case_insensitive {
            let (start, len) = ir.offsets[0];
            let start = start as usize;
            let end = start + len as usize;
            let needle = ir.packed_bytes.get(start..end).ok_or_else(|| {
                Error::PatternCompilationFailed {
                    reason: "single-literal scanner references bytes outside the packed pattern buffer. Fix: rebuild the pattern set and retry."
                        .to_string(),
                }
            })?;
            let pattern_id = ir.literal_automaton_ids.first().copied().ok_or_else(|| {
                Error::PatternCompilationFailed {
                    reason: "single-literal scanner is missing an automaton-to-pattern mapping. Fix: rebuild the pattern set and retry."
                        .to_string(),
                }
            })?;
            // pattern_id comes from automaton ID which is bounded by pattern count
            #[allow(clippy::cast_possible_truncation)]
            let pattern_id = pattern_id as u32;

            return Ok(Self {
                inner: CachedScannerInner::SingleLiteral {
                    finder: memchr::memmem::Finder::new(needle),
                    pattern_id,
                    needle_len: len,
                },
            });
        }
        if should_use_hash_scanner(ir) {
            return Ok(Self {
                inner: CachedScannerInner::Hash {
                    scanner: HashScanner::build(ir),
                },
            });
        }

        Ok(Self {
            inner: CachedScannerInner::Generic { ir },
        })
    }

    /// Scan a payload using the cached scanner state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InputTooLarge`] if `data.len()` exceeds 4 GiB.
    pub fn scan(&self, data: &[u8], out_matches: &mut [Match]) -> Result<usize> {
        check_input_size(data)?;

        match &self.inner {
            CachedScannerInner::SingleLiteral {
                finder,
                pattern_id,
                needle_len,
            } => self::scan::scan_single_literal_with_finder(
                data,
                finder,
                *pattern_id,
                *needle_len,
                out_matches,
            ),
            CachedScannerInner::Hash { scanner } => {
                let mut count = 0;
                self::scan::scan_literals_fast_with_hash(scanner, data, &mut |matched| {
                    if count >= out_matches.len() {
                        false
                    } else {
                        out_matches[count] = matched;
                        count += 1;
                        true
                    }
                })?;
                sort_matches_if_needed(&mut out_matches[..count]);
                if count == out_matches.len() {
                    return Err(Error::MatchBufferOverflow {
                        count,
                        max: count.min(MAX_CPU_MATCHES),
                    });
                }
                Ok(count)
            }
            CachedScannerInner::Generic { ir } => scan(ir, data, out_matches),
        }
    }
}

pub(crate) fn should_use_hash_scanner(ir: &PatternIR) -> bool {
    !ir.case_insensitive
        && ir.regex_dfas.is_empty()
        && ir.literal_automaton_ids.len() > HASH_SCANNER_LITERAL_THRESHOLD
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::{PatternSet, Result};

    use super::{scan, Match, PatternIR, MAX_CPU_MATCHES};

    fn legacy_sort_matches(matches: &mut [Match]) {
        matches.sort_unstable_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then(left.pattern_id.cmp(&right.pattern_id))
                .then(left.end.cmp(&right.end))
        });
    }

    fn legacy_scan(ir: &PatternIR, data: &[u8], out_matches: &mut [Match]) -> Result<usize> {
        if ir.offsets.len() == 1 && ir.regex_dfas.is_empty() && !ir.case_insensitive {
            let (start, len) = ir.offsets[0];
            let needle = &ir.packed_bytes[start as usize..(start + len) as usize];
            let finder = memchr::memmem::Finder::new(needle);
            // pattern_id comes from automaton ID which is bounded by pattern count
            #[allow(clippy::cast_possible_truncation)]
            let pattern_id = ir.literal_automaton_ids.first().copied().unwrap_or(0) as u32;
            let needle_len = needle.len() as u32;
            let mut count = 0;
            for pos in finder.find_iter(data) {
                if count >= out_matches.len() {
                    return Err(crate::Error::MatchBufferOverflow {
                        count: MAX_CPU_MATCHES,
                        max: MAX_CPU_MATCHES,
                    });
                }
                // SAFETY: pos < data.len() and check_input_size ensures data.len() <= u32::MAX
                out_matches[count] = Match {
                    pattern_id,
                    start: pos as u32,
                    end: pos as u32 + needle_len,
                    padding: 0,
                };
                count += 1;
            }
            return Ok(count);
        }

        let Some(ac) = &ir.literal_automaton else {
            return Ok(0);
        };

        let mut count = 0;
        for mat in ac.find_iter(data) {
            if count >= out_matches.len() {
                return Err(crate::Error::MatchBufferOverflow {
                    count: MAX_CPU_MATCHES,
                    max: MAX_CPU_MATCHES,
                });
            }
            // pattern_id comes from automaton ID which is bounded by pattern count
            #[allow(clippy::cast_possible_truncation)]
            let pattern_id = ir.literal_automaton_ids[mat.pattern().as_usize()] as u32;
            // mat indices are bounded by data.len() which is validated <= u32::MAX
            out_matches[count] = Match {
                pattern_id,
                start: mat.start() as u32,
                end: mat.end() as u32,
                padding: 0,
            };
            count += 1;
        }
        for dfa in &ir.regex_dfas {
            count += dfa.scan_native(data, &mut out_matches[count..])?;
        }
        legacy_sort_matches(&mut out_matches[..count]);
        Ok(count)
    }

    #[test]
    fn single_literal() {
        let ps = PatternSet::builder().literal("hello").build().unwrap();
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let res = ps.scan(b"say hello world").unwrap();
        let n = res.len();
        for (i, m) in res.into_iter().enumerate() {
            if i < matches.len() {
                matches[i] = m;
            }
        }
        assert_eq!(n, 1);
        assert_eq!(matches[0].pattern_id, 0);
        assert_eq!(matches[0].start, 4);
        assert_eq!(matches[0].end, 9);
    }

    #[test]
    fn regex_and_literal_patterns_share_a_result_set() {
        let ps = PatternSet::builder()
            .literal("cat")
            .regex(r"d.g")
            .build()
            .unwrap();
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let res = ps.scan(b"cat dog").unwrap();
        let n = res.len();
        for (i, m) in res.into_iter().enumerate() {
            if i < matches.len() {
                matches[i] = m;
            }
        }
        assert_eq!(n, 2);
        assert!(matches[..n].iter().any(|m| m.pattern_id == 0));
        assert!(matches[..n].iter().any(|m| m.pattern_id == 1));
    }

    #[test]
    fn non_overlapping_repeated_matches() {
        let ps = PatternSet::builder().literal("aa").build().unwrap();
        // Non-overlapping: "aaaa" → matches at [0,2) and [2,4) = 2 matches
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let res = ps.scan(b"aaaa").unwrap();
        let n = res.len();
        for (i, m) in res.into_iter().enumerate() {
            if i < matches.len() {
                matches[i] = m;
            }
        }
        assert_eq!(n, 2);
    }

    #[test]
    fn overlapping_repeated_matches() {
        let ps = PatternSet::builder().literal("aa").build().unwrap();
        // Overlapping: "aaaa" → matches at [0,2), [1,3), [2,4) = 3 matches
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let res = ps.scan_overlapping(b"aaaa").unwrap();
        let n = res.len();
        for (i, m) in res.into_iter().enumerate() {
            if i < matches.len() {
                matches[i] = m;
            }
        }
        assert_eq!(n, 3);
    }

    #[test]
    fn scan_and_scan_overlapping_agree_on_non_overlapping_input() {
        let ps = PatternSet::builder()
            .literal("hello")
            .literal("world")
            .build()
            .unwrap();
        let data = b"hello world";
        let mut fast = [Match::from_parts(0, 0, 0); 10];
        let mut full = [Match::from_parts(0, 0, 0); 10];
        let res = ps.scan(data).unwrap();
        let nf = res.len();
        for (i, m) in res.into_iter().enumerate() {
            if i < fast.len() {
                fast[i] = m;
            }
        }
        let res = ps.scan_overlapping(data).unwrap();
        let nl = res.len();
        for (i, m) in res.into_iter().enumerate() {
            if i < full.len() {
                full[i] = m;
            }
        }
        // When no overlaps exist, both modes produce the same matches
        assert_eq!(nf, nl);
        assert_eq!(&fast[..nf], &full[..nl]);
    }

    #[test]
    fn cached_scanner_matches_standard_scan() {
        let patterns = PatternSet::builder().literal("needle").build().unwrap();
        let scanner = super::CachedScanner::new(patterns.ir()).unwrap();
        let data = b"needle and another needle";
        let mut cached_matches = [Match::from_parts(0, 0, 0); 10];
        let mut standard_matches = [Match::from_parts(0, 0, 0); 10];
        let nc = scanner.scan(data, &mut cached_matches).unwrap();
        let res = patterns.scan(data).unwrap();
        let ns = res.len();
        for (i, m) in res.into_iter().enumerate() {
            if i < standard_matches.len() {
                standard_matches[i] = m;
            }
        }
        assert_eq!(nc, ns);
        assert_eq!(&cached_matches[..nc], &standard_matches[..ns]);
    }

    #[test]
    #[ignore = "manual benchmark"]
    fn benchmark_pattern_scale_100mb() {
        const FILE_SIZE: usize = 100 * 1_048_576;
        const STRIDE: usize = 1_024;

        fn make_pattern(index: usize) -> String {
            format!("p{index:03}match")
        }

        fn make_data(needle: &[u8]) -> Vec<u8> {
            let mut data = vec![b'x'; FILE_SIZE];
            let last_start = FILE_SIZE.saturating_sub(needle.len());
            for offset in (0..=last_start).step_by(STRIDE) {
                data[offset..offset + needle.len()].copy_from_slice(needle);
            }
            data
        }

        let data = make_data(make_pattern(0).as_bytes());

        for count in [1, 10, 100] {
            let mut builder = PatternSet::builder();
            for index in 0..count {
                builder = builder.literal(&make_pattern(index));
            }
            let patterns = builder.build().unwrap();

            let mut legacy_matches_buf = vec![Match::from_parts(0, 0, 0); FILE_SIZE / STRIDE + 100];
            let mut matches_buf = vec![Match::from_parts(0, 0, 0); FILE_SIZE / STRIDE + 100];

            let legacy_started = Instant::now();
            let legacy_matches_count =
                legacy_scan(patterns.ir(), &data, &mut legacy_matches_buf).unwrap();
            let legacy_elapsed = legacy_started.elapsed();

            let optimized_started = Instant::now();
            let matches_count = scan(patterns.ir(), &data, &mut matches_buf).unwrap();
            let optimized_elapsed = optimized_started.elapsed();

            assert_eq!(matches_count, FILE_SIZE / STRIDE);
            assert_eq!(
                &legacy_matches_buf[..legacy_matches_count],
                &matches_buf[..matches_count]
            );
            println!(
                "pattern_scale_100mb count={count} matches={} legacy_ms={:.3} optimized_ms={:.3} speedup={:.2}x",
                matches_count,
                legacy_elapsed.as_secs_f64() * 1_000.0,
                optimized_elapsed.as_secs_f64() * 1_000.0,
                legacy_elapsed.as_secs_f64() / optimized_elapsed.as_secs_f64()
            );
        }
    }
}

#[cfg(test)]
mod p0_tests {
    use super::*;
    use crate::PatternSet;

    #[test]
    fn test_p0_pattern_not_lost_at_scale() {
        let mut builder = PatternSet::builder();
        builder = builder.literal("burpcollaborator.net");
        builder = builder.literal("burpcollaborator%2Enet");
        for i in 0..50 {
            builder = builder.literal_bytes(format!("dummy_pattern_{i}_filler").into_bytes());
        }
        let ps = builder.build().unwrap();
        let input = b"var domain = \"burpcollaborator%2Enet\";";
        let matches = ps.scan(input).unwrap();
        let found_urlencode = matches.iter().any(|m| {
            let matched = &input[m.start as usize..m.end as usize];
            matched == b"burpcollaborator%2Enet"
        });
        assert!(
            found_urlencode,
            "P0: burpcollaborator%2Enet not found in matches: {:?}",
            matches
        );
    }
}
