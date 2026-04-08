use crate::dfa::RegexDFA;
use crate::literal_prefilter::LiteralPrefilterTable;

/// GPU-friendly intermediate representation for literals and regex DFAs.
#[derive(Debug, Clone)]
pub struct PatternIR {
    /// All literal bytes packed contiguously.
    pub packed_bytes: Vec<u8>,
    /// `(start_offset, length)` for each literal pattern in `packed_bytes`.
    pub offsets: Vec<(u32, u32)>,
    /// Optional names for each pattern in insertion order.
    pub names: Vec<Option<String>>,
    pub(crate) matchers: Vec<CompiledPattern>,
    pub(crate) regex_dfas: Vec<RegexDFA>,
    /// Regex patterns in insertion order, paired with their original PatternSet index.
    pub(crate) regex_patterns: Vec<(usize, String)>,
    pub(crate) max_pattern_len: usize,
    pub(crate) hash_window_len: u32,
    pub(crate) literal_prefilter_table: LiteralPrefilterTable,
    pub(crate) literal_automaton: Option<aho_corasick::AhoCorasick>,
    pub(crate) literal_automaton_ids: Vec<usize>,
    /// True when the literal automaton uses case-insensitive matching.
    /// Disables the memchr fast path (memchr doesn't support case folding).
    pub(crate) case_insensitive: bool,
    /// Cached regex for fast case-insensitive literal scanning via SIMD.
    pub(crate) fast_ci_regex: Option<regex::bytes::Regex>,
    /// Pre-built hash scanner for large literal sets (>1000 patterns).
    /// Built once at PatternSet construction instead of per-scan.
    pub(crate) cached_hash_scanner: Option<crate::hash_scan::HashScanner>,
}

impl PatternIR {
    /// Return the compiled DFA backends for pure-regex patterns.
    pub fn regex_dfas(&self) -> &[crate::dfa::RegexDFA] {
        &self.regex_dfas
    }

    /// Return regex patterns paired with their original PatternSet index.
    ///
    /// Used by consumers (e.g., warpscan) that need to reconstruct a unified
    /// pattern set merging patterns from multiple sources.
    pub fn regex_patterns(&self) -> &[(usize, String)] {
        &self.regex_patterns
    }

    /// Return the byte length of the longest compiled pattern.
    #[must_use]
    pub fn max_pattern_len(&self) -> usize {
        self.max_pattern_len
    }
}

/// One compiled pattern entry in insertion order.
#[derive(Debug, Clone)]
pub struct CompiledPattern {
    pub(crate) id: usize,
    pub(crate) kind: CompiledPatternKind,
}

/// The backend representation for a compiled pattern.
#[derive(Debug, Clone)]
pub enum CompiledPatternKind {
    Literal { literal_index: usize },
    Regex,
}

#[derive(Debug)]
pub(super) enum PendingPattern {
    Literal {
        bytes: PendingLiteralBytes,
        name: Option<String>,
    },
    Regex {
        pattern: String,
        name: Option<String>,
    },
}

#[derive(Debug)]
pub(super) enum PendingLiteralBytes {
    Utf8(String),
    Raw(Vec<u8>),
}

impl PendingLiteralBytes {
    pub(super) fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Utf8(pattern) => pattern.as_bytes(),
            Self::Raw(bytes) => bytes,
        }
    }
}
