//! Pattern-set specialization — selects optimal scan strategy based on pattern characteristics.
//!
//! This module analyzes a compiled pattern set and selects the fastest available
//! scanning algorithm for its specific characteristics.

use crate::error::Result;
use crate::pattern::{PatternIR, PatternSet};
use crate::Match;

/// Maximum length for SingleMemchr optimization.
const SINGLE_MEMCHR_MAX_LEN: usize = 32;

/// Maximum number of literals for MultiMemchr optimization.
const MULTI_MEMCHR_MAX_PATTERNS: usize = 8;

/// Maximum length for each literal in MultiMemchr optimization.
const MULTI_MEMCHR_MAX_LEN: usize = 16;

/// Scanning strategy selected based on pattern set characteristics.
#[derive(Clone)]
pub enum ScanStrategy {
    /// Single literal pattern, length <= 32 bytes.
    /// Uses memchr::memmem for maximum speed.
    SingleMemchr {
        /// The needle bytes to search for.
        needle: Vec<u8>,
        /// Original pattern ID.
        pattern_id: u32,
    },

    /// 2-8 literal patterns, all <= 16 bytes.
    /// Uses memchr::memmem for each pattern.
    MultiMemchr {
        /// The needles to search for, with their pattern IDs.
        needles: Vec<(Vec<u8>, u32)>,
    },

    /// Many literals (current default) — uses Aho-Corasick.
    AhoCorasick,

    /// Single regex pattern — uses regex crate's optimized SIMD engine.
    /// Compiled regex is cached for reuse across scans.
    SingleRegex {
        /// The regex pattern string.
        pattern: String,
        /// Original pattern ID.
        pattern_id: u32,
        /// Cached compiled regex — initialized on first use.
        compiled: std::sync::OnceLock<regex::bytes::Regex>,
    },

    /// Complex patterns requiring full DFA evaluation.
    FullDfa,
}

impl std::fmt::Debug for ScanStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingleMemchr { needle, pattern_id } => f
                .debug_struct("SingleMemchr")
                .field("needle_len", &needle.len())
                .field("pattern_id", pattern_id)
                .finish(),
            Self::MultiMemchr { needles } => f
                .debug_struct("MultiMemchr")
                .field("count", &needles.len())
                .finish(),
            Self::AhoCorasick => write!(f, "AhoCorasick"),
            Self::SingleRegex {
                pattern,
                pattern_id,
                ..
            } => f
                .debug_struct("SingleRegex")
                .field("pattern", pattern)
                .field("pattern_id", pattern_id)
                .field("cached", &self.is_regex_cached())
                .finish(),
            Self::FullDfa => write!(f, "FullDfa"),
        }
    }
}

impl ScanStrategy {
    /// Check if the regex has been compiled and cached.
    fn is_regex_cached(&self) -> bool {
        matches!(self, Self::SingleRegex { compiled, .. } if compiled.get().is_some())
    }
}

impl ScanStrategy {
    /// Analyze a pattern set and select the optimal scanning strategy.
    ///
    /// Selection rules (in priority order):
    /// 1. SingleMemchr: 1 literal, len <= 32 bytes, case-sensitive
    /// 2. MultiMemchr: 2-8 literals, all <= 16 bytes, case-sensitive, no regex
    /// 3. SingleRegex: 1 regex pattern, no literals
    /// 4. FullDfa: mixed patterns with regex
    /// 5. AhoCorasick: default for many literals or complex cases
    #[inline]
    pub fn select(patterns: &PatternSet) -> Self {
        let ir = &patterns.ir;

        // Case-insensitive patterns can't use memchr fast paths
        if ir.case_insensitive {
            if ir.regex_dfas.is_empty() {
                return Self::AhoCorasick;
            }
            return Self::FullDfa;
        }

        let literal_count = ir.offsets.len();
        let regex_count = ir.regex_patterns.len();

        // Check if all literals meet MultiMemchr length requirements
        let all_literals_short = ir.offsets.iter().all(|(_start, len)| {
            let len = *len as usize;
            len <= MULTI_MEMCHR_MAX_LEN
        });

        // Single literal fast path
        if literal_count == 1 && regex_count == 0 {
            let (start, len) = ir.offsets[0];
            let len = len as usize;

            if len <= SINGLE_MEMCHR_MAX_LEN {
                // SAFETY: start is bounded by packed_bytes.len() which fits in usize
                #[allow(clippy::cast_possible_truncation)]
                let needle = ir.packed_bytes[start as usize..(start as usize + len)].to_vec();
                // pattern_id is bounded by the number of patterns
                #[allow(clippy::cast_possible_truncation)]
                let pattern_id = ir.literal_automaton_ids.first().copied().unwrap_or(0) as u32;
                return Self::SingleMemchr { needle, pattern_id };
            }
        }

        // Multi literal fast path (2-8 patterns, all short)
        if (2..=MULTI_MEMCHR_MAX_PATTERNS).contains(&literal_count)
            && regex_count == 0
            && all_literals_short
        {
            let mut needles = Vec::with_capacity(literal_count);
            for (i, (start, len)) in ir.offsets.iter().enumerate() {
                let len = *len as usize;
                // SAFETY: start is bounded by packed_bytes.len() which fits in usize
                #[allow(clippy::cast_possible_truncation)]
                let needle = ir.packed_bytes[*start as usize..(*start as usize + len)].to_vec();
                // pattern_id is bounded by the number of patterns
                #[allow(clippy::cast_possible_truncation)]
                let pattern_id = ir.literal_automaton_ids.get(i).copied().unwrap_or(i) as u32;
                needles.push((needle, pattern_id));
            }
            return Self::MultiMemchr { needles };
        }

        // Single regex fast path
        if literal_count == 0 && regex_count == 1 {
            let (pattern_id, pattern) = &ir.regex_patterns[0];
            // pattern_id is bounded by the number of patterns
            #[allow(clippy::cast_possible_truncation)]
            let pattern_id_u32 = *pattern_id as u32;
            return Self::SingleRegex {
                pattern: pattern.clone(),
                pattern_id: pattern_id_u32,
                compiled: std::sync::OnceLock::new(),
            };
        }

        // Mixed patterns with regex require full DFA
        if !ir.regex_dfas.is_empty() {
            return Self::FullDfa;
        }

        // Default: Aho-Corasick for many literals
        Self::AhoCorasick
    }

    /// Execute a scan using the selected strategy.
    ///
    /// Returns matches in the same format as the standard scan, ensuring
    /// parity between all strategies.
    #[inline]
    pub fn scan(&self, data: &[u8], ir: &PatternIR, out_matches: &mut [Match]) -> Result<usize> {
        match self {
            Self::SingleMemchr { needle, pattern_id } => {
                scan_single_memchr(data, needle, *pattern_id, out_matches)
            }
            Self::MultiMemchr { needles } => scan_multi_memchr(data, needles, out_matches),
            Self::AhoCorasick => crate::cpu::scan(ir, data, out_matches),
            Self::SingleRegex {
                pattern,
                pattern_id,
                compiled,
            } => scan_single_regex(data, pattern, *pattern_id, compiled, out_matches),
            Self::FullDfa => crate::cpu::scan(ir, data, out_matches),
        }
    }

    /// Execute a streaming scan using the selected strategy.
    ///
    /// The visitor callback receives matches as they are found. Return `false`
    /// to stop scanning early.
    #[inline]
    pub fn scan_with<F>(&self, data: &[u8], ir: &PatternIR, visitor: &mut F) -> Result<()>
    where
        F: FnMut(Match) -> bool,
    {
        match self {
            Self::SingleMemchr { needle, pattern_id } => {
                scan_single_memchr_with(data, needle, *pattern_id, visitor)
            }
            Self::MultiMemchr { needles } => scan_multi_memchr_with(data, needles, visitor),
            Self::AhoCorasick => crate::cpu::scan_with(ir, data, visitor),
            Self::SingleRegex {
                pattern,
                pattern_id,
                compiled,
            } => scan_single_regex_with(data, pattern, *pattern_id, compiled, visitor),
            Self::FullDfa => crate::cpu::scan_with(ir, data, visitor),
        }
    }
}

#[inline]
fn scan_single_memchr(
    data: &[u8],
    needle: &[u8],
    pattern_id: u32,
    out_matches: &mut [Match],
) -> Result<usize> {
    crate::cpu::check_input_size(data)?;
    // needle_len is bounded by SINGLE_MEMCHR_MAX_LEN (32 bytes)
    #[allow(clippy::cast_possible_truncation)]
    let needle_len = needle.len() as u32;
    let finder = memchr::memmem::Finder::new(needle);

    let mut count = 0;
    for pos in finder.find_iter(data) {
        if count >= out_matches.len() {
            return Err(crate::Error::MatchBufferOverflow {
                count,
                max: out_matches.len().min(crate::cpu::MAX_CPU_MATCHES),
            });
        }
        // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = pos as u32;
        #[allow(clippy::cast_possible_truncation)]
        let end = (pos as u32).saturating_add(needle_len);
        out_matches[count] = Match {
            pattern_id,
            start,
            end,
            padding: 0,
        };
        count += 1;
    }
    Ok(count)
}

/// Streaming scan for a single literal.
#[inline]
fn scan_single_memchr_with<F>(
    data: &[u8],
    needle: &[u8],
    pattern_id: u32,
    visitor: &mut F,
) -> Result<()>
where
    F: FnMut(Match) -> bool,
{
    crate::cpu::check_input_size(data)?;
    // needle_len is bounded by SINGLE_MEMCHR_MAX_LEN (32 bytes)
    #[allow(clippy::cast_possible_truncation)]
    let needle_len = needle.len() as u32;
    let finder = memchr::memmem::Finder::new(needle);
    let mut count = 0usize;

    for pos in finder.find_iter(data) {
        if count >= crate::cpu::MAX_CPU_MATCHES {
            return Err(crate::Error::MatchBufferOverflow {
                count,
                max: crate::cpu::MAX_CPU_MATCHES,
            });
        }
        count += 1;
        // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = pos as u32;
        #[allow(clippy::cast_possible_truncation)]
        let end = (pos as u32).saturating_add(needle_len);
        if !visitor(Match {
            pattern_id,
            start,
            end,
            padding: 0,
        }) {
            break;
        }
    }

    Ok(())
}

#[inline]
fn scan_multi_memchr(
    data: &[u8],
    needles: &[(Vec<u8>, u32)],
    out_matches: &mut [Match],
) -> Result<usize> {
    crate::cpu::check_input_size(data)?;
    // Collect all matches, sort them, then deduplicate overlaps.
    // To do this strictly in-place we'd need complex iterators.
    // Since this is a specialized fast-path for short patterns,
    // we allocate a small intermediate array just for sorting.
    let mut matches = Vec::with_capacity(estimate_match_capacity(data.len()));

    for (needle, pattern_id) in needles {
        let finder = memchr::memmem::Finder::new(needle);
        // needle_len is bounded by MULTI_MEMCHR_MAX_LEN (16 bytes)
        #[allow(clippy::cast_possible_truncation)]
        let needle_len = needle.len() as u32;

        for pos in finder.find_iter(data) {
            // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
            #[allow(clippy::cast_possible_truncation)]
            let start = pos as u32;
            #[allow(clippy::cast_possible_truncation)]
            let end = (pos as u32).saturating_add(needle_len);
            matches.push(Match {
                pattern_id: *pattern_id,
                start,
                end,
                padding: 0,
            });
        }
    }

    matches.sort_unstable_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.pattern_id.cmp(&b.pattern_id))
            .then(a.end.cmp(&b.end))
    });

    let mut last_end = 0;
    let mut count = 0;
    for m in matches {
        if m.start >= last_end {
            if count >= out_matches.len() {
                return Err(crate::Error::MatchBufferOverflow {
                    count,
                    max: out_matches.len().min(crate::cpu::MAX_CPU_MATCHES),
                });
            }
            last_end = m.end;
            out_matches[count] = m;
            count += 1;
        }
    }
    Ok(count)
}

/// Streaming scan for multiple literals.
#[inline]
fn scan_multi_memchr_with<F>(data: &[u8], needles: &[(Vec<u8>, u32)], visitor: &mut F) -> Result<()>
where
    F: FnMut(Match) -> bool,
{
    crate::cpu::check_input_size(data)?;
    // Collect all matches, sort them, then dispatch
    // For true streaming with early termination, we'd need to merge streams
    // For simplicity and correctness, we collect and sort
    let mut matches = Vec::with_capacity(estimate_match_capacity(data.len()));

    for (needle, pattern_id) in needles {
        let finder = memchr::memmem::Finder::new(needle);
        // needle_len is bounded by MULTI_MEMCHR_MAX_LEN (16 bytes)
        #[allow(clippy::cast_possible_truncation)]
        let needle_len = needle.len() as u32;

        for pos in finder.find_iter(data) {
            // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
            #[allow(clippy::cast_possible_truncation)]
            let start = pos as u32;
            #[allow(clippy::cast_possible_truncation)]
            let end = (pos as u32).saturating_add(needle_len);
            matches.push(Match {
                pattern_id: *pattern_id,
                start,
                end,
                padding: 0,
            });
        }
    }

    // Sort by position, then pattern_id for deterministic output
    matches.sort_unstable_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.pattern_id.cmp(&b.pattern_id))
            .then(a.end.cmp(&b.end))
    });

    let mut count = 0usize;
    for m in matches {
        if count >= crate::cpu::MAX_CPU_MATCHES {
            return Err(crate::Error::MatchBufferOverflow {
                count,
                max: crate::cpu::MAX_CPU_MATCHES,
            });
        }
        count += 1;
        if !visitor(m) {
            break;
        }
    }

    Ok(())
}

#[inline]
fn scan_single_regex(
    data: &[u8],
    pattern: &str,
    pattern_id: u32,
    compiled: &std::sync::OnceLock<regex::bytes::Regex>,
    out_matches: &mut [Match],
) -> Result<usize> {
    crate::cpu::check_input_size(data)?;

    // Get or compile the regex - compilation happens only once
    let re = compiled.get_or_init(|| {
        // Invalid regex patterns compile to a regex that matches nothing
        regex::bytes::Regex::new(pattern).unwrap_or_else(|_| {
            #[allow(clippy::expect_used)]
            let r = regex::bytes::Regex::new("a^").expect("valid regex"); // Never matches
            r
        })
    });

    let mut count = 0;
    let mut last_end = 0;
    for m in re.find_iter(data) {
        // SAFETY: m.start() < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let m_start = m.start() as u32;
        if m_start >= last_end {
            if count >= out_matches.len() {
                return Err(crate::Error::MatchBufferOverflow {
                    count,
                    max: out_matches.len().min(crate::cpu::MAX_CPU_MATCHES),
                });
            }
            // SAFETY: m indices are bounded by data.len() which is validated <= u32::MAX
            #[allow(clippy::cast_possible_truncation)]
            let start = m.start() as u32;
            #[allow(clippy::cast_possible_truncation)]
            let end = m.end() as u32;
            out_matches[count] = Match {
                pattern_id,
                start,
                end,
                padding: 0,
            };
            last_end = out_matches[count].end;
            count += 1;
        }
    }
    Ok(count)
}

/// Streaming scan for a single regex.
#[inline]
fn scan_single_regex_with<F>(
    data: &[u8],
    pattern: &str,
    pattern_id: u32,
    compiled: &std::sync::OnceLock<regex::bytes::Regex>,
    visitor: &mut F,
) -> Result<()>
where
    F: FnMut(Match) -> bool,
{
    crate::cpu::check_input_size(data)?;

    // Get or compile the regex - compilation happens only once
    let re = compiled.get_or_init(|| {
        // Invalid regex patterns compile to a regex that matches nothing
        regex::bytes::Regex::new(pattern).unwrap_or_else(|_| {
            #[allow(clippy::expect_used)]
            let r = regex::bytes::Regex::new("a^").expect("valid regex"); // Never matches
            r
        })
    });

    let mut count = 0usize;
    for m in re.find_iter(data) {
        if count >= crate::cpu::MAX_CPU_MATCHES {
            return Err(crate::Error::MatchBufferOverflow {
                count,
                max: crate::cpu::MAX_CPU_MATCHES,
            });
        }
        count += 1;
        // SAFETY: m indices are bounded by data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = m.start() as u32;
        #[allow(clippy::cast_possible_truncation)]
        let end = m.end() as u32;
        if !visitor(Match {
            pattern_id,
            start,
            end,
            padding: 0,
        }) {
            break;
        }
    }

    Ok(())
}

/// Estimate initial match capacity based on data size.
///
/// Uses `data_len / 4` as a conservative estimate — in the worst case, a
/// 2-byte pattern matches every other byte. For inputs > 4MB, caps at 1M
/// matches to avoid pathological allocations.
#[inline]
pub(crate) fn estimate_match_capacity(data_len: usize) -> usize {
    data_len.clamp(64, 1_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_memchr_strategy_selected() {
        let patterns = PatternSet::builder().literal("needle").build().unwrap();
        let strategy = ScanStrategy::select(&patterns);
        assert!(
            matches!(strategy, ScanStrategy::SingleMemchr { .. }),
            "Expected SingleMemchr for single short literal, got {:?}",
            strategy
        );
    }

    #[test]
    fn multi_memchr_strategy_selected() {
        let patterns = PatternSet::builder()
            .literal("foo")
            .literal("bar")
            .literal("baz")
            .build()
            .unwrap();
        let strategy = ScanStrategy::select(&patterns);
        assert!(
            matches!(strategy, ScanStrategy::MultiMemchr { .. }),
            "Expected MultiMemchr for 3 short literals, got {:?}",
            strategy
        );
    }

    #[test]
    fn aho_corasick_strategy_for_many_literals() {
        let mut builder = PatternSet::builder();
        for i in 0..10 {
            builder = builder.literal(&format!("pat{i}"));
        }
        let patterns = builder.build().unwrap();
        let strategy = ScanStrategy::select(&patterns);
        assert!(
            matches!(strategy, ScanStrategy::AhoCorasick),
            "Expected AhoCorasick for 10 literals, got {:?}",
            strategy
        );
    }

    #[test]
    fn aho_corasick_strategy_for_long_literal() {
        let long_needle = "a".repeat(100);
        let patterns = PatternSet::builder().literal(&long_needle).build().unwrap();
        let strategy = ScanStrategy::select(&patterns);
        assert!(
            matches!(strategy, ScanStrategy::AhoCorasick),
            "Expected AhoCorasick for long literal, got {:?}",
            strategy
        );
    }

    #[test]
    fn single_regex_strategy_selected() {
        let patterns = PatternSet::builder().regex(r"[a-z]+").build().unwrap();
        let strategy = ScanStrategy::select(&patterns);
        assert!(
            matches!(strategy, ScanStrategy::SingleRegex { .. }),
            "Expected SingleRegex for single regex, got {:?}",
            strategy
        );
    }

    #[test]
    fn full_dfa_strategy_for_mixed_patterns() {
        let patterns = PatternSet::builder()
            .literal("needle")
            .regex(r"[a-z]+")
            .build()
            .unwrap();
        let strategy = ScanStrategy::select(&patterns);
        assert!(
            matches!(strategy, ScanStrategy::FullDfa),
            "Expected FullDfa for mixed patterns, got {:?}",
            strategy
        );
    }

    #[test]
    fn case_insensitive_uses_aho_corasick() {
        let patterns = PatternSet::builder()
            .literal("needle")
            .case_insensitive(true)
            .build()
            .unwrap();
        let strategy = ScanStrategy::select(&patterns);
        assert!(
            matches!(strategy, ScanStrategy::AhoCorasick),
            "Expected AhoCorasick for case-insensitive, got {:?}",
            strategy
        );
    }

    #[test]
    fn single_memchr_finds_matches() {
        let data = b"needle in a haystack with needle";
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let n = scan_single_memchr(data, b"needle", 42, &mut matches).unwrap();
        assert_eq!(n, 2);
        assert_eq!(matches[0].pattern_id, 42);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[0].end, 6);
        assert_eq!(matches[1].start, 26);
        assert_eq!(matches[1].end, 32);
    }

    #[test]
    fn multi_memchr_finds_all_patterns() {
        let data = b"foo bar baz foo";
        let needles = vec![
            (b"foo".to_vec(), 0u32),
            (b"bar".to_vec(), 1u32),
            (b"baz".to_vec(), 2u32),
        ];
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let n = scan_multi_memchr(data, &needles, &mut matches).unwrap();
        // foo at 0, bar at 4, baz at 8, foo at 12
        assert_eq!(n, 4);
        assert_eq!(matches[0].pattern_id, 0);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[1].pattern_id, 1);
        assert_eq!(matches[1].start, 4);
        assert_eq!(matches[2].pattern_id, 2);
        assert_eq!(matches[2].start, 8);
        assert_eq!(matches[3].pattern_id, 0);
        assert_eq!(matches[3].start, 12);
    }

    #[test]
    fn single_regex_finds_matches() {
        let data = b"abc123def456";
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let n = scan_single_regex(
            data,
            r"[0-9]+",
            99,
            &std::sync::OnceLock::new(),
            &mut matches,
        )
        .unwrap();
        assert_eq!(n, 2);
        assert_eq!(matches[0].pattern_id, 99);
        assert_eq!(matches[0].start, 3);
        assert_eq!(matches[0].end, 6);
        assert_eq!(matches[1].start, 9);
        assert_eq!(matches[1].end, 12);
    }

    #[test]
    fn empty_data_returns_no_matches() {
        let data = b"";
        let mut matches = [Match::from_parts(0, 0, 0); 10];
        let n = scan_single_memchr(data, b"needle", 0, &mut matches).unwrap();
        assert_eq!(n, 0);

        let n = scan_multi_memchr(data, &[(b"foo".to_vec(), 0)], &mut matches).unwrap();
        assert_eq!(n, 0);

        let n =
            scan_single_regex(data, r".", 0, &std::sync::OnceLock::new(), &mut matches).unwrap();
        assert_eq!(n, 0);
    }
}
