use regex_syntax::hir::HirKind;

use super::ir::{
    CompiledPattern, CompiledPatternKind, PatternIR, PendingLiteralBytes, PendingPattern,
};
use super::PatternSet;
use crate::dfa::RegexDFA;
use crate::error::{Error, Result};
use crate::literal_prefilter::LiteralPrefilterTable;

/// Builder for constructing a [`PatternSet`].
#[derive(Debug, Default)]
pub struct PatternSetBuilder {
    pub(super) patterns: Vec<PendingPattern>,
    /// When true, literal patterns use ASCII case-insensitive matching.
    pub(super) ascii_case_insensitive: bool,
}

impl PatternSetBuilder {
    /// Add a literal pattern. Accepts owned String to avoid copying when caller already owns the data.
    pub fn literal(mut self, pattern: impl Into<String>) -> Self {
        self.patterns.push(PendingPattern::Literal {
            bytes: PendingLiteralBytes::Utf8(pattern.into()),
            name: None,
        });
        self
    }

    /// Add a literal pattern with raw bytes. Accepts owned Vec to avoid copying.
    pub fn literal_bytes(mut self, pattern: impl Into<Vec<u8>>) -> Self {
        self.patterns.push(PendingPattern::Literal {
            bytes: PendingLiteralBytes::Raw(pattern.into()),
            name: None,
        });
        self
    }

    /// Add a named literal pattern.
    pub fn named_literal(mut self, name: &str, pattern: &str) -> Self {
        self.patterns.push(PendingPattern::Literal {
            bytes: PendingLiteralBytes::Utf8(pattern.to_owned()),
            name: Some(name.to_string()),
        });
        self
    }

    /// Add a regex pattern.
    pub fn regex(mut self, pattern: &str) -> Self {
        self.patterns.push(PendingPattern::Regex {
            pattern: pattern.to_string(),
            name: None,
        });
        self
    }

    /// Enable ASCII case-insensitive matching for literal patterns.
    ///
    /// When set, the Aho-Corasick automaton folds `a`-`z` to `A`-`Z`
    /// during both index building and searching. This is ~10x faster
    /// than wrapping literals in `(?i:...)` regex because it avoids
    /// the DFA compilation and backward-start search overhead.
    pub fn case_insensitive(mut self, yes: bool) -> Self {
        self.ascii_case_insensitive = yes;
        self
    }

    /// Add a named regex pattern.
    pub fn named_regex(mut self, name: &str, pattern: &str) -> Self {
        self.patterns.push(PendingPattern::Regex {
            pattern: pattern.to_string(),
            name: Some(name.to_string()),
        });
        self
    }

    /// Compile all patterns into a [`PatternSet`].
    #[allow(clippy::too_many_lines)]
    pub fn build(self) -> Result<PatternSet> {
        if self.patterns.is_empty() {
            return Err(Error::EmptyPatternSet);
        }

        let max_u32 = u32::MAX as usize;
        if self.patterns.len() > max_u32 {
            return Err(Error::PatternSetTooLarge {
                patterns: self.patterns.len(),
                bytes: 0,
                max_bytes: max_u32,
            });
        }

        let mut packed_bytes = Vec::new();
        let mut offsets = Vec::new();
        let mut names = Vec::new();
        let mut matchers = Vec::with_capacity(self.patterns.len());
        let mut regex_dfas = Vec::new();
        let mut literal_hashes = Vec::new();
        let mut max_pattern_len = 0usize;
        let hash_window_len = 8u32;

        let mut regex_patterns = Vec::new();
        let mut regex_original_ids = Vec::new();

        for (i, pattern) in self.patterns.into_iter().enumerate() {
            match pattern {
                PendingPattern::Literal { bytes, name } => {
                    let bytes = bytes.as_bytes();
                    if bytes.is_empty() {
                        return Err(Error::EmptyPattern { index: i });
                    }
                    if bytes.len() > max_u32 {
                        return Err(Error::PatternTooLarge {
                            index: i,
                            bytes: bytes.len(),
                            max: max_u32,
                        });
                    }
                    let start = u32::try_from(packed_bytes.len()).map_err(|_| {
                        Error::PatternSetTooLarge {
                            patterns: matchers.len() + 1,
                            bytes: packed_bytes.len(),
                            max_bytes: max_u32,
                        }
                    })?;
                    let len = u32::try_from(bytes.len()).map_err(|_| Error::PatternTooLarge {
                        index: i,
                        bytes: bytes.len(),
                        max: max_u32,
                    })?;
                    literal_hashes.push(literal_prefilter_hash(bytes, hash_window_len as usize));
                    max_pattern_len = max_pattern_len.max(bytes.len());
                    packed_bytes.extend_from_slice(bytes);
                    offsets.push((start, len));
                    names.push(name);
                    matchers.push(CompiledPattern {
                        id: i,
                        kind: CompiledPatternKind::Literal {
                            literal_index: offsets.len() - 1,
                        },
                    });
                }
                PendingPattern::Regex { pattern, name } => {
                    if pattern.is_empty() {
                        return Err(Error::EmptyPattern { index: i });
                    }
                    // Parse HIR once for pathological detection before DFA compilation.
                    // Use byte-level (non-Unicode) parsing so that patterns like
                    // \x90 (non-ASCII bytes) and [\x00-\xff] wildcards are accepted.
                    let hir = regex_syntax::ParserBuilder::new()
                        .unicode(false)
                        .utf8(false)
                        .build()
                        .parse(&pattern)
                        .map_err(|e| Error::PatternCompilationFailed {
                            reason: format!("invalid regex at index {i}: {e}"),
                        })?;
                    if is_pathological(&hir) {
                        return Err(Error::PathologicalRegex { index: i });
                    }

                    // Optimization: if the regex is just a literal (no metacharacters,
                    // no flags), compile it as a literal for memchr fast path.
                    if let Some(literal_bytes) = extract_literal_from_hir(&hir) {
                        if !literal_bytes.is_empty() {
                            let start = u32::try_from(packed_bytes.len()).map_err(|_| {
                                Error::PatternSetTooLarge {
                                    patterns: matchers.len() + 1,
                                    bytes: packed_bytes.len(),
                                    max_bytes: max_u32,
                                }
                            })?;
                            let len = u32::try_from(literal_bytes.len()).map_err(|_| {
                                Error::PatternTooLarge {
                                    index: i,
                                    bytes: literal_bytes.len(),
                                    max: max_u32,
                                }
                            })?;
                            literal_hashes.push(literal_prefilter_hash(
                                &literal_bytes,
                                hash_window_len as usize,
                            ));
                            max_pattern_len = max_pattern_len.max(literal_bytes.len());
                            packed_bytes.extend_from_slice(&literal_bytes);
                            offsets.push((start, len));
                            names.push(name);
                            matchers.push(CompiledPattern {
                                id: i,
                                kind: CompiledPatternKind::Literal {
                                    literal_index: offsets.len() - 1,
                                },
                            });
                            continue;
                        }
                    }

                    regex_patterns.push((i, pattern));
                    regex_original_ids.push(i);
                    names.push(name);
                    matchers.push(CompiledPattern {
                        id: i,
                        kind: CompiledPatternKind::Regex,
                    });
                }
            }
        }

        if !regex_patterns.is_empty() {
            let str_refs: Vec<&str> = regex_patterns
                .iter()
                .map(|(_, pattern)| AsRef::as_ref(pattern))
                .collect();
            regex_dfas.push(RegexDFA::build(&str_refs, &regex_original_ids)?);
        }

        let mut literal_automaton_ids = Vec::new();
        let mut literal_patterns = Vec::new();
        for matcher in &matchers {
            if let CompiledPatternKind::Literal { literal_index } = matcher.kind {
                let (start, len) = offsets[literal_index];
                literal_patterns.push(&packed_bytes[start as usize..(start + len) as usize]);
                literal_automaton_ids.push(matcher.id);
            }
        }
        let literal_automaton = if literal_patterns.is_empty() {
            None
        } else if self.ascii_case_insensitive {
            Some(
                aho_corasick::AhoCorasick::builder()
                    .match_kind(aho_corasick::MatchKind::LeftmostFirst)
                    .ascii_case_insensitive(true)
                    .build(&literal_patterns)
                    .map_err(|error| Error::PatternCompilationFailed {
                        reason: error.to_string(),
                    })?,
            )
        } else {
            Some(
                aho_corasick::AhoCorasick::builder()
                    // Use LeftmostLongest to ensure all patterns are found.
                    // LeftmostFirst can silently skip patterns when shorter patterns
                    // at the same position shadow longer ones in a large pattern set.
                    .match_kind(aho_corasick::MatchKind::LeftmostFirst)
                    .build(&literal_patterns)
                    .map_err(|error| Error::PatternCompilationFailed {
                        reason: error.to_string(),
                    })?,
            )
        };

        let literal_prefilter_table =
            LiteralPrefilterTable::build(&offsets, &literal_hashes, hash_window_len).map_err(
                |reason| Error::PatternCompilationFailed {
                    reason: format!(
                        "failed to build literal prefilter hash table: {reason}. Fix: reduce the literal set size or lower the configured hash window."
                    ),
                },
            )?;

        // Build cached CI regex before moving values into PatternIR
        let fast_ci_regex =
            if self.ascii_case_insensitive && offsets.len() == 1 && regex_dfas.is_empty() {
                let (start, len) = offsets[0];
                let needle = &packed_bytes[start as usize..(start + len) as usize];
                std::str::from_utf8(needle).ok().and_then(|s| {
                    regex::bytes::Regex::new(&format!("(?i:{})", regex_syntax::escape(s))).ok()
                })
            } else {
                None
            };

        // Pre-build hash scanner for large literal sets (avoids per-scan rebuild).
        let ir = PatternIR {
            packed_bytes,
            offsets,
            names,
            regex_patterns,
            matchers,
            regex_dfas,
            max_pattern_len,
            hash_window_len,
            literal_prefilter_table,
            literal_automaton,
            case_insensitive: self.ascii_case_insensitive,
            literal_automaton_ids,
            fast_ci_regex,
            cached_hash_scanner: None,
        };
        let cached_hash_scanner = if crate::cpu::should_use_hash_scanner(&ir) {
            Some(crate::hash_scan::HashScanner::build(&ir))
        } else {
            None
        };
        let ir = PatternIR {
            cached_hash_scanner,
            ..ir
        };
        let mut ps = PatternSet {
            strategy: crate::specialize::ScanStrategy::AhoCorasick, // overwritten by select() below
            ir,
        };
        ps.strategy = crate::specialize::ScanStrategy::select(&ps);
        Ok(ps)
    }
}

pub fn literal_prefilter_hash(bytes: &[u8], window_len: usize) -> u32 {
    let mut hash = 2_166_136_261u32;
    for &byte in bytes.iter().take(window_len.max(1)) {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

/// Try to extract a literal byte string from a regex HIR.
///
/// Returns `Some(bytes)` if the regex is equivalent to a simple literal
/// (a concatenation of single-byte literals with no flags, anchors, or
/// metacharacters). Returns `None` for anything more complex.
///
/// This enables the memchr fast path for regex patterns like "needle" that
/// are really just literal strings, giving ~10x speedup over DFA scan.
fn extract_literal_from_hir(hir: &regex_syntax::hir::Hir) -> Option<Vec<u8>> {
    match hir.kind() {
        HirKind::Literal(lit) => Some(lit.0.to_vec()),
        HirKind::Concat(list) => {
            let mut bytes = Vec::new();
            for item in list {
                bytes.extend(extract_literal_from_hir(item)?);
            }
            Some(bytes)
        }
        // Non-capturing groups and captures with a single literal child
        // are transparent — extract through them.
        HirKind::Capture(cap) => extract_literal_from_hir(&cap.sub),
        _ => None,
    }
}

/// Check regex HIR for nested unbounded repetitions that cause catastrophic
/// DFA state explosion (e.g. `(a+)+`, `(a*)*`, `([0-9]+)+`).
///
/// Returns `true` if the pattern is pathological and should be rejected.
fn is_pathological(hir: &regex_syntax::hir::Hir) -> bool {
    fn has_rep(h: &regex_syntax::hir::Hir) -> bool {
        match h.kind() {
            HirKind::Repetition(_) => true,
            HirKind::Concat(list) | HirKind::Alternation(list) => list.iter().any(has_rep),
            HirKind::Capture(cap) => has_rep(&cap.sub),
            _ => false,
        }
    }

    match hir.kind() {
        HirKind::Repetition(rep) => {
            // Only pathological if BOTH outer and inner are unbounded.
            // (x+)? is fine (outer bounded to 0-1). (x+)+ is pathological.
            let outer_unbounded = rep.max.is_none() || rep.max.is_some_and(|m| m > 1);
            if outer_unbounded && has_rep(&rep.sub) {
                return true;
            }
            is_pathological(&rep.sub)
        }
        HirKind::Concat(list) | HirKind::Alternation(list) => list.iter().any(is_pathological),
        HirKind::Capture(cap) => is_pathological(&cap.sub),
        _ => false,
    }
}
