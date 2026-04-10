use aho_corasick::AhoCorasick;

use super::{check_input_size, sort_matches_if_needed, MAX_CPU_INPUT_BYTES, MAX_CPU_MATCHES};
use crate::error::{Error, Result};
use crate::hash_scan::HashScanner;
use crate::pattern::{CompiledPatternKind, PatternIR};
use crate::Match;

/// Non-overlapping scan — SIMD-accelerated Teddy prefilters enabled.
///
/// Uses `find_iter` which enables Aho-Corasick's vectorized Teddy prefilter
/// (128-bit or 256-bit SIMD). First match per position wins. This is the
/// default for CLI output and matches ripgrep's semantics.
///
/// # Errors
///
/// Returns [`Error::InputTooLarge`] if `data.len()` exceeds 4 GiB.
#[inline]
pub fn scan(ir: &PatternIR, data: &[u8], out_matches: &mut [Match]) -> Result<usize> {
    check_input_size(data)?;

    let mut count = 0;
    scan_with(ir, data, &mut |matched| {
        if count < out_matches.len() {
            out_matches[count] = matched;
            count += 1;
            true
        } else {
            false
        }
    })?;
    sort_matches_if_needed(&mut out_matches[..count]);
    if count == out_matches.len() {
        return Err(Error::MatchBufferOverflow {
            count,
            max: out_matches.len().min(MAX_CPU_MATCHES),
        });
    }
    Ok(count)
}

/// Non-overlapping scan that streams matches into `visitor`.
///
/// The callback receives matches in backend emission order. Return `false` to
/// stop scanning early.
#[inline]
pub fn scan_with<F>(ir: &PatternIR, data: &[u8], visitor: &mut F) -> Result<()>
where
    F: FnMut(Match) -> bool,
{
    check_input_size(data)?;

    if ir.offsets.len() == 1 && ir.regex_dfas.is_empty() && !ir.case_insensitive {
        let (start, len) = ir.offsets[0];
        let needle = &ir.packed_bytes[start as usize..(start + len) as usize];
        return scan_single_literal_with(ir, data, needle, visitor);
    }
    // Fast path for case-insensitive single literals: use regex crate's SIMD
    // engine via (?i:pattern) instead of the slower Aho-Corasick CI automaton.
    // NOTE: regex is compiled per-call here. For daemon/streaming use,
    // prefer CachedScanner which compiles once and reuses across files.
    if ir.offsets.len() == 1 && ir.regex_dfas.is_empty() && ir.case_insensitive {
        if let Some(re) = &ir.fast_ci_regex {
            let pattern_id = ir.literal_automaton_ids.first().copied().ok_or_else(|| {
                Error::PatternCompilationFailed {
                    reason: "case-insensitive single-literal scan is missing automaton-to-pattern mapping. Fix: rebuild the pattern set."
                        .to_string(),
                }
            })?;
            let pattern_id = u32::try_from(pattern_id).map_err(|_| Error::PatternCompilationFailed {
                reason: "case-insensitive single-literal pattern ID exceeds u32::MAX. Fix: rebuild the pattern set.".to_string(),
            })?;
            for m in re.find_iter(data) {
                let s = u32::try_from(m.start()).map_err(|_| Error::InputTooLarge {
                    bytes: m.start(),
                    max_bytes: MAX_CPU_INPUT_BYTES,
                })?;
                let e = u32::try_from(m.end()).map_err(|_| Error::InputTooLarge {
                    bytes: m.end(),
                    max_bytes: MAX_CPU_INPUT_BYTES,
                })?;
                if !visitor(Match {
                    pattern_id,
                    start: s,
                    end: e,
                    padding: 0,
                }) {
                    return Ok(());
                }
            }
            return Ok(());
        }
    }
    if let Some(scanner) = &ir.cached_hash_scanner {
        scan_literals_fast_with_hash(scanner, data, visitor)?;
    } else if super::should_use_hash_scanner(ir) {
        let scanner = HashScanner::build(ir)?;
        scan_literals_fast_with_hash(&scanner, data, visitor)?;
    } else {
        scan_literals_fast_with(ir, data, visitor)?;
    }

    // Always scan regex DFAs — they are independent of the literal scanner.
    for dfa in &ir.regex_dfas {
        dfa.scan_native_with(data, visitor)?;
    }
    Ok(())
}

/// Count non-overlapping matches without allocating a `Vec<Match>`.
#[inline]
pub fn scan_count(ir: &PatternIR, data: &[u8]) -> Result<usize> {
    let mut count = 0usize;
    scan_with(ir, data, &mut |_| {
        count += 1;
        true
    })?;
    Ok(count)
}

/// Single-literal scan using memchr::memmem — fastest possible path.
#[inline]
pub(crate) fn scan_single_literal_with_finder(
    data: &[u8],
    finder: &memchr::memmem::Finder<'_>,
    pattern_id: u32,
    needle_len: u32,
    out_matches: &mut [Match],
) -> Result<usize> {
    let mut count = 0;
    for pos in finder.find_iter(data) {
        if count >= out_matches.len() {
            return Err(Error::MatchBufferOverflow {
                count,
                max: out_matches.len().min(MAX_CPU_MATCHES),
            });
        }
        // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = pos as u32;
        // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
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

#[inline]
fn scan_single_literal_with<F>(
    ir: &PatternIR,
    data: &[u8],
    needle: &[u8],
    visitor: &mut F,
) -> Result<()>
where
    F: FnMut(Match) -> bool,
{
    let finder = memchr::memmem::Finder::new(needle);
    let pattern_id = ir.literal_automaton_ids.first().copied().ok_or_else(|| {
        Error::PatternCompilationFailed {
            reason: "single-literal scan is missing automaton-to-pattern mapping. Fix: rebuild the pattern set."
                .to_string(),
        }
    })?;
    let pattern_id = u32::try_from(pattern_id).map_err(|_| Error::PatternCompilationFailed {
        reason: "single-literal pattern ID exceeds u32::MAX. Fix: rebuild the pattern set.".to_string(),
    })?;
    // needle_len is bounded by the literal length which is reasonable
    #[allow(clippy::cast_possible_truncation)]
    let needle_len = needle.len() as u32;
    for (count, pos) in finder.find_iter(data).enumerate() {
        if count >= MAX_CPU_MATCHES {
            return Err(Error::MatchBufferOverflow {
                count: MAX_CPU_MATCHES,
                max: MAX_CPU_MATCHES,
            });
        }
        // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = pos as u32;
        // SAFETY: pos < data.len() which is validated <= u32::MAX by check_input_size
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
pub(crate) fn scan_literals_fast_with_hash<F>(
    scanner: &HashScanner,
    data: &[u8],
    visitor: &mut F,
) -> Result<()>
where
    F: FnMut(Match) -> bool,
{
    for (count, mat) in scanner.scan(data).into_iter().enumerate() {
        if count >= MAX_CPU_MATCHES {
            return Err(Error::MatchBufferOverflow {
                count: MAX_CPU_MATCHES,
                max: MAX_CPU_MATCHES,
            });
        }
        if !visitor(mat) {
            break;
        }
    }

    Ok(())
}

/// Overlapping scan — reports every match at every byte position.
///
/// Uses `find_overlapping_iter` which disables Teddy SIMD prefilters but
/// guarantees every overlapping match is reported. Required for GPU backend
/// parity (GPU threads inspect every byte independently).
///
/// # Errors
///
/// Returns [`Error::InputTooLarge`] if `data.len()` exceeds 4 GiB.
#[inline]
pub fn scan_overlapping(ir: &PatternIR, data: &[u8], out_matches: &mut [Match]) -> Result<usize> {
    check_input_size(data)?;
    let mut count = scan_literals_overlapping(ir, data, out_matches)?;
    for dfa in &ir.regex_dfas {
        // Overlapping mode also uses native scan for correctness.
        // True overlapping regex is not yet supported — this uses
        // non-overlapping native search as a baseline.
        let added = dfa.scan_native(data, &mut out_matches[count..])?;
        count += added;
    }
    sort_matches_if_needed(&mut out_matches[..count]);
    Ok(count)
}

#[inline]
pub(crate) fn scan_literals_fast_with<F>(ir: &PatternIR, data: &[u8], visitor: &mut F) -> Result<()>
where
    F: FnMut(Match) -> bool,
{
    let Some(ac) = &ir.literal_automaton else {
        return Ok(());
    };

    for (count, mat) in ac.find_iter(data).enumerate() {
        if count >= MAX_CPU_MATCHES {
            return Err(Error::MatchBufferOverflow {
                count: MAX_CPU_MATCHES,
                max: MAX_CPU_MATCHES,
            });
        }
        let pattern_id = ir
            .literal_automaton_ids
            .get(mat.pattern().as_usize())
            .copied()
            .and_then(|pattern_id| u32::try_from(pattern_id).ok())
            .ok_or_else(|| {
                Error::PatternCompilationFailed {
                    reason: format!(
                        "literal AC pattern mapping missing or out of range for index {}. Fix: rebuild the pattern set.",
                        mat.pattern().as_usize()
                    ),
                }
            })?;
        // SAFETY: mat.start() < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = mat.start() as u32;
        // SAFETY: mat.end() <= data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let end = mat.end() as u32;
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

/// Overlapping literal scan — visits every byte position.
///
/// Rebuilds the Aho-Corasick automaton with `MatchKind::Standard` because
/// the primary automaton uses `LeftmostFirst` (for Teddy SIMD prefilters)
/// which does not support `find_overlapping_iter`.
#[inline]
fn scan_literals_overlapping(
    ir: &PatternIR,
    data: &[u8],
    out_matches: &mut [Match],
) -> Result<usize> {
    if ir.literal_automaton.is_none() {
        return Ok(0);
    }

    // Rebuild patterns in the exact order used by the primary Aho-Corasick automaton
    // and preserve the corresponding original pattern IDs.
    let mut literal_patterns: Vec<&[u8]> = Vec::with_capacity(ir.literal_automaton_ids.len());
    let mut literal_pattern_ids = Vec::with_capacity(ir.literal_automaton_ids.len());
    for matcher in &ir.matchers {
        let CompiledPatternKind::Literal { literal_index } = matcher.kind else {
            continue;
        };
        let (start_u32, len_u32) = *ir.offsets.get(literal_index).ok_or_else(|| {
            Error::PatternCompilationFailed {
                reason:
                    "literal automaton mapping references an invalid literal index. Fix: rebuild the pattern set."
                        .to_string(),
            }
        })?;
        let pattern_id = ir.literal_automaton_ids.get(literal_index).copied().ok_or_else(|| {
            Error::PatternCompilationFailed {
                reason:
                    "literal automaton IDs are missing entries. Fix: rebuild the pattern set."
                        .to_string(),
            }
        })?;
        let start = usize::try_from(start_u32).map_err(|_| Error::PatternCompilationFailed {
            reason: "literal pattern start offset does not fit in usize. Fix: rebuild the pattern set."
                .to_string(),
        })?;
        let len = usize::try_from(len_u32).map_err(|_| Error::PatternCompilationFailed {
            reason: "literal pattern length does not fit in usize. Fix: rebuild the pattern set."
                .to_string(),
        })?;
        let end = start.checked_add(len).ok_or_else(|| Error::PatternCompilationFailed {
            reason: "literal pattern range overflow. Fix: rebuild the pattern set.".to_string(),
        })?;
        if end > ir.packed_bytes.len() {
            return Err(Error::PatternCompilationFailed {
                reason: "literal pattern references bytes outside the packed buffer. Fix: rebuild the pattern set."
                    .to_string(),
            });
        }
        literal_patterns.push(&ir.packed_bytes[start..end]);
        literal_pattern_ids.push(pattern_id);
    }
    if literal_patterns.is_empty() {
        return Ok(0);
    }

    let ac = aho_corasick::AhoCorasick::builder()
        .match_kind(aho_corasick::MatchKind::Standard)
        .ascii_case_insensitive(ir.case_insensitive)
        .build(&literal_patterns)
        .map_err(|error| Error::PatternCompilationFailed {
            reason: error.to_string(),
        })?;

    let mut count = 0;
    for mat in ac.find_overlapping_iter(data) {
        if count >= out_matches.len() {
            return Err(Error::MatchBufferOverflow {
                count,
                max: out_matches.len().min(MAX_CPU_MATCHES),
            });
        }
        let pattern_id = literal_pattern_ids
            .get(mat.pattern().as_usize())
            .copied()
            .ok_or_else(|| {
                Error::PatternCompilationFailed {
                    reason: format!(
                        "overlapping literal mapping missing for automaton index {}. Fix: rebuild pattern set.",
                        mat.pattern().as_usize()
                    ),
                }
            })
            .and_then(|raw_id| {
                u32::try_from(raw_id).map_err(|_| Error::PatternCompilationFailed {
                    reason: "literal pattern ID exceeds u32::MAX. Fix: rebuild the pattern set."
                        .to_string(),
                })
            })?;
        // SAFETY: mat.start() < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = mat.start() as u32;
        // SAFETY: mat.end() <= data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let end = mat.end() as u32;
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

/// Directly scan an arbitrary payload using the provided AhoCorasick automaton.
///
/// Uses non-overlapping mode for SIMD acceleration.
#[inline]
pub fn scan_aho_corasick(
    aho: &AhoCorasick,
    data: &[u8],
    out_matches: &mut [Match],
) -> Result<usize> {
    let mut count = 0;
    for mat in aho.find_iter(data) {
        if count >= out_matches.len() {
            return Err(Error::MatchBufferOverflow {
                count,
                max: out_matches.len().min(MAX_CPU_MATCHES),
            });
        }
        // SAFETY: pattern index is valid by AhoCorasick contract
        #[allow(clippy::cast_possible_truncation)]
        let pattern_id = mat.pattern().as_usize() as u32;
        // SAFETY: mat.start() < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = mat.start() as u32;
        // SAFETY: mat.end() <= data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let end = mat.end() as u32;
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

/// Directly scan with overlapping semantics using the provided AhoCorasick automaton.
#[inline]
pub fn scan_aho_corasick_overlapping(
    aho: &AhoCorasick,
    data: &[u8],
    out_matches: &mut [Match],
) -> Result<usize> {
    let mut count = 0;
    for mat in aho.find_overlapping_iter(data) {
        if count >= out_matches.len() {
            return Err(Error::MatchBufferOverflow {
                count,
                max: out_matches.len().min(MAX_CPU_MATCHES),
            });
        }
        // SAFETY: pattern index is valid by AhoCorasick contract
        #[allow(clippy::cast_possible_truncation)]
        let pattern_id = mat.pattern().as_usize() as u32;
        // SAFETY: mat.start() < data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let start = mat.start() as u32;
        // SAFETY: mat.end() <= data.len() which is validated <= u32::MAX by check_input_size
        #[allow(clippy::cast_possible_truncation)]
        let end = mat.end() as u32;
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
