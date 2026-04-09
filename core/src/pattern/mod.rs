//! Pattern compilation — the IR that all backends consume.

use arc_swap::ArcSwap;
use std::borrow::Cow;
use std::sync::Arc;

use crate::dfa::RegexDFA;
use crate::error::{Error, Result};
use crate::specialize::estimate_match_capacity;

pub mod compiler;
pub mod ir;

pub use compiler::{literal_prefilter_hash, PatternSetBuilder};
pub use ir::{CompiledPatternKind, PatternIR};

/// A thread-safe, lock-free updatable pattern set.
pub struct HotSwapPatternSet {
    current: Arc<ArcSwap<PatternSet>>,
}

impl HotSwapPatternSet {
    /// Construct from an already-compiled [`PatternSet`].
    pub fn new(pattern_set: PatternSet) -> Self {
        Self::from_arc(Arc::new(pattern_set))
    }

    /// Construct from a shared [`PatternSet`] reference.
    pub fn from_arc(pattern_set: Arc<PatternSet>) -> Self {
        Self {
            current: Arc::new(ArcSwap::from(pattern_set)),
        }
    }

    /// Atomically replace the active pattern set.
    #[must_use]
    pub fn swap(&self, next: PatternSet) -> bool {
        self.swap_arc(Arc::new(next))
    }

    /// Atomically replace the active pattern set from an `Arc` handle.
    #[must_use]
    pub fn swap_arc(&self, next: Arc<PatternSet>) -> bool {
        let current = self.current.load_full();
        if Arc::ptr_eq(&current, &next) {
            return false;
        }
        self.current.store(next);
        true
    }

    fn current(&self) -> Arc<PatternSet> {
        self.current.load_full()
    }

    /// Same contract as [`PatternSet::scan`], using the currently active pattern set.
    pub fn scan(&self, data: &[u8], out_matches: &mut [crate::Match]) -> Result<usize> {
        let current = self.current();
        crate::cpu::check_input_size(data)?;
        let mut count = 0;
        crate::cpu::scan_with(&current.ir, data, &mut |matched| {
            if count >= out_matches.len() {
                false
            } else {
                out_matches[count] = matched;
                count += 1;
                true
            }
        })?;
        crate::cpu::sort_matches_if_needed(&mut out_matches[..count]);
        if count == out_matches.len() {
            return Err(Error::MatchBufferOverflow {
                count,
                max: count.min(crate::cpu::MAX_CPU_MATCHES),
            });
        }
        Ok(count)
    }

    /// Same contract as [`PatternSet::scan_with`], using the currently active pattern set.
    pub fn scan_with<F>(&self, data: &[u8], mut visitor: F) -> Result<()>
    where
        F: FnMut(crate::Match) -> bool,
    {
        let current = self.current();
        crate::cpu::scan_with(&current.ir, data, &mut visitor)
    }
}

/// A compiled set of patterns ready for scanning.
#[derive(Debug, Clone)]
pub struct PatternSet {
    pub(crate) ir: PatternIR,
    /// Cached scan strategy — computed once at construction, reused for every scan.
    pub(crate) strategy: crate::specialize::ScanStrategy,
}

impl PatternSet {
    /// Start building a new pattern set.
    pub fn builder() -> PatternSetBuilder {
        PatternSetBuilder::default()
    }

    /// Number of patterns in the set.
    pub fn len(&self) -> usize {
        self.ir.matchers.len()
    }

    /// Whether the pattern set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ir.matchers.is_empty()
    }

    /// Non-overlapping scan — SIMD-accelerated via Teddy prefilters.
    ///
    /// First match per position wins. This is the default for CLI output
    /// and matches ripgrep's semantics. **40x faster** than overlapping
    /// mode on multi-literal pattern sets.
    ///
    /// # Fast Path Specialization
    ///
    /// This method automatically selects the optimal scanning strategy based on
    /// the pattern set characteristics:
    /// - Single short literal (<=32 bytes): uses memchr::memmem
    /// - 2-8 short literals (all <=16 bytes): uses memchr for each pattern
    /// - Single regex: uses regex crate's optimized SIMD engine
    /// - Mixed patterns: uses DFA evaluation
    /// - Many literals: uses Aho-Corasick
    pub fn scan(&self, data: &[u8]) -> Result<Vec<crate::Match>> {
        let mut matches =
            vec![crate::Match::from_parts(0, 0, 0); estimate_match_capacity(data.len())];
        let n = self.scan_to_buffer(data, &mut matches)?;
        matches.truncate(n);
        Ok(matches)
    }

    /// Run the optimized scan strategy using a provided match buffer.
    pub fn scan_to_buffer(&self, data: &[u8], out_matches: &mut [crate::Match]) -> Result<usize> {
        self.strategy.scan(data, &self.ir, out_matches)
    }

    /// Visit non-overlapping matches without materializing a `Vec<Match>`.
    ///
    /// This is intended for streaming consumers such as count-only search
    /// paths that need match offsets but do not need to retain every match.
    ///
    /// ```rust
    /// use warpstate::PatternSet;
    ///
    /// let patterns = PatternSet::builder().literal("needle").build()?;
    /// let mut count = 0usize;
    /// patterns.scan_with(b"needle in a needle stack", |matched| {
    ///     let _ = matched;
    ///     count += 1;
    ///     true
    /// })?;
    /// assert_eq!(count, 2);
    /// # Ok::<(), warpstate::Error>(())
    /// ```
    pub fn scan_with<F>(&self, data: &[u8], mut visitor: F) -> Result<()>
    where
        F: FnMut(crate::Match) -> bool,
    {
        let strategy = crate::specialize::ScanStrategy::select(self);
        strategy.scan_with(data, &self.ir, &mut visitor)
    }

    /// Count non-overlapping matches without allocating a `Vec<Match>`.
    ///
    /// ```rust
    /// use warpstate::PatternSet;
    ///
    /// let patterns = PatternSet::builder().literal("needle").build()?;
    /// assert_eq!(patterns.scan_count(b"needle in a needle stack")?, 2);
    /// # Ok::<(), warpstate::Error>(())
    /// ```
    pub fn scan_count(&self, data: &[u8]) -> Result<usize> {
        crate::cpu::scan_count(&self.ir, data)
    }

    /// Overlapping scan — reports every match at every byte position.
    ///
    /// Disables Teddy SIMD prefilters. Required for GPU backend parity
    /// testing (GPU threads inspect every byte independently).
    pub fn scan_overlapping(&self, data: &[u8]) -> Result<Vec<crate::Match>> {
        let mut matches =
            vec![crate::Match::from_parts(0, 0, 0); estimate_match_capacity(data.len())];
        let n = self.scan_overlapping_to_buffer(data, &mut matches)?;
        matches.truncate(n);
        Ok(matches)
    }

    /// Overlapping scan using a provided match buffer.
    pub fn scan_overlapping_to_buffer(
        &self,
        data: &[u8],
        out_matches: &mut [crate::Match],
    ) -> Result<usize> {
        crate::cpu::scan_overlapping(&self.ir, data, out_matches)
    }

    /// Access the compiled IR.
    pub fn ir(&self) -> &PatternIR {
        &self.ir
    }

    /// Return the byte length of the longest compiled pattern.
    #[must_use]
    pub fn max_pattern_len(&self) -> usize {
        self.ir.max_pattern_len()
    }

    pub(crate) fn compiled_regex_dfa(&self) -> Result<Cow<'_, RegexDFA>> {
        let mut all_patterns = Vec::new();
        let mut original_ids = Vec::new();

        for matcher in &self.ir.matchers {
            if let CompiledPatternKind::Literal { literal_index } = matcher.kind {
                let (start, len) = self.ir.offsets[literal_index];
                let bytes = &self.ir.packed_bytes[start as usize..(start + len) as usize];
                let literal = std::str::from_utf8(bytes).map_err(|_| Error::PatternCompilationFailed {
                    reason: format!(
                        "pattern {} contains non-UTF-8 bytes and cannot be lowered into a regex DFA backend. Fix: use the literal GPU backend for raw byte patterns.",
                        matcher.id
                    ),
                })?;
                all_patterns.push(regex::escape(literal));
                original_ids.push(matcher.id);
            }
        }

        for (id, pattern) in &self.ir.regex_patterns {
            all_patterns.push(pattern.clone());
            original_ids.push(*id);
        }

        if all_patterns.is_empty() {
            return Err(Error::PatternCompilationFailed {
                reason: "no patterns available to build DFA".to_string(),
            });
        }

        // Fast path: if there are no literals, the pre-built regex DFA is exactly what we need.
        // We ensure it has all patterns by checking if the IDs match exactly.
        if let Some(dfa) = self.ir.regex_dfas.first() {
            if original_ids.len() == dfa.native_original_ids.len() {
                return Ok(Cow::Borrowed(dfa));
            }
        }

        let pattern_refs: Vec<&str> = all_patterns.iter().map(String::as_str).collect();
        RegexDFA::build(&pattern_refs, &original_ids).map(Cow::Owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::compiler::literal_prefilter_hash;

    #[test]
    fn builder_supports_literal_and_regex_patterns() {
        let set = PatternSet::builder()
            .literal("password")
            .regex(r"s[e3]cret")
            .build()
            .unwrap();

        assert_eq!(set.len(), 2);
        assert_eq!(set.ir.offsets.len(), 1);
        assert_eq!(set.ir.regex_dfas.len(), 1);
    }

    // === Adversarial Pattern Tests ===

    /// Test that empty pattern set returns error
    #[test]
    fn pattern_empty_set_error() {
        let result = PatternSet::builder().build();
        assert!(matches!(result, Err(Error::EmptyPatternSet)));
    }

    /// Test that single pattern succeeds
    #[test]
    fn pattern_single_literal_success() {
        let set = PatternSet::builder().literal("test").build().unwrap();
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
    }

    /// Test empty literal pattern returns error
    #[test]
    fn pattern_empty_literal_error() {
        let result = PatternSet::builder().literal("").build();
        assert!(matches!(result, Err(Error::EmptyPattern { index: 0 })));
    }

    /// Test empty regex pattern returns error
    #[test]
    fn pattern_empty_regex_error() {
        let result = PatternSet::builder().regex("").build();
        assert!(matches!(result, Err(Error::EmptyPattern { index: 0 })));
    }

    /// Test empty literal bytes returns error
    #[test]
    fn pattern_empty_literal_bytes_error() {
        let result = PatternSet::builder().literal_bytes(b"").build();
        assert!(matches!(result, Err(Error::EmptyPattern { index: 0 })));
    }

    /// Test named literal pattern
    #[test]
    fn pattern_named_literal() {
        let set = PatternSet::builder()
            .named_literal("my_pattern", "test")
            .build()
            .unwrap();

        assert_eq!(set.len(), 1);
        assert_eq!(set.ir.names[0], Some("my_pattern".to_string()));
    }

    /// Test named regex pattern
    #[test]
    fn pattern_named_regex() {
        let set = PatternSet::builder()
            .named_regex("my_regex", r"t+es*t")
            .build()
            .unwrap();

        assert_eq!(set.len(), 1);
        assert_eq!(set.ir.names[0], Some("my_regex".to_string()));
    }

    /// Test pattern count edge case: single pattern
    #[test]
    fn pattern_single_count() {
        let set = PatternSet::builder().literal("a").build().unwrap();
        assert_eq!(set.len(), 1);
        assert_eq!(set.ir.matchers.len(), 1);
    }

    /// Test many patterns
    #[test]
    fn pattern_many_literals() {
        let mut builder = PatternSet::builder();
        for i in 0..100 {
            builder = builder.literal(&format!("pattern_{i}"));
        }
        let set = builder.build().unwrap();
        assert_eq!(set.len(), 100);
    }

    /// Test literal_bytes with binary data
    #[test]
    fn pattern_literal_bytes_binary() {
        let binary = vec![0x00, 0xFF, 0x42, 0x13, 0x37];
        let set = PatternSet::builder()
            .literal_bytes(binary.clone())
            .build()
            .unwrap();

        assert_eq!(set.len(), 1);
        assert_eq!(set.ir.packed_bytes, binary);
    }

    /// Test mixed patterns (literals and regex)
    #[test]
    fn pattern_mixed_types() {
        let set = PatternSet::builder()
            .literal("literal1")
            .regex(r"regex[0-9]+")
            .literal("literal2")
            .regex(r"another.*pattern")
            .build()
            .unwrap();

        assert_eq!(set.len(), 4);
        // Should have 2 literals and 2 regex patterns
        assert_eq!(set.ir.offsets.len(), 2); // 2 literals
        assert_eq!(set.ir.regex_dfas.len(), 1); // 2 regexes in one DFA
    }

    /// Test PatternSet is_empty
    #[test]
    fn pattern_is_empty() {
        let set = PatternSet::builder().literal("test").build().unwrap();
        assert!(!set.is_empty());
    }

    /// Test pattern IR access
    #[test]
    fn pattern_ir_access() {
        let set = PatternSet::builder().literal("test").build().unwrap();
        let ir = set.ir();
        assert_eq!(ir.offsets.len(), 1);
        assert_eq!(ir.packed_bytes, b"test");
    }

    /// Test hash computation for patterns
    #[test]
    fn pattern_prefilter_hash() {
        let hash1 = literal_prefilter_hash(b"hello", 8);
        let hash2 = literal_prefilter_hash(b"hello", 8);
        let hash3 = literal_prefilter_hash(b"world", 8);

        assert_eq!(hash1, hash2); // Same input = same hash
        assert_ne!(hash1, hash3); // Different input = different hash
    }

    /// Test hash with window smaller than pattern
    #[test]
    fn pattern_prefilter_hash_short_window() {
        let hash = literal_prefilter_hash(b"longer_pattern", 4);
        // Should only hash first 4 bytes
        let hash_4bytes = literal_prefilter_hash(b"long", 4);
        assert_eq!(hash, hash_4bytes);
    }

    /// Test hash with window larger than pattern
    #[test]
    fn pattern_prefilter_hash_long_window() {
        let hash = literal_prefilter_hash(b"ab", 10);
        // Window longer than pattern, should hash all bytes
        let hash_full = literal_prefilter_hash(b"ab", 2);
        assert_eq!(hash, hash_full);
    }

    #[test]
    fn literal_prefilter_table_probes_match_linear_scan() {
        let set = PatternSet::builder()
            .literal("a")
            .literal("ab")
            .literal("alphabet")
            .literal("beta")
            .literal("betamax")
            .build()
            .unwrap();

        for (literal_index, &(start, len)) in set.ir.offsets.iter().enumerate() {
            let prefix_len = len.min(set.ir.hash_window_len).max(1);
            let literal =
                &set.ir.packed_bytes[start as usize..(start.saturating_add(len)) as usize];
            let hash = literal_prefilter_hash(literal, set.ir.hash_window_len as usize);
            let hits: Vec<u32> = set
                .ir
                .literal_prefilter_table
                .probe(prefix_len, hash)
                .collect();
            assert!(hits.contains(&(literal_index as u32)));
        }
    }

    /// Nested repetitions like `(a+)+` are rejected.
    #[test]
    fn pathological_regex_rejected() {
        let result = PatternSet::builder().regex(r"(a+)+").build();
        assert!(matches!(result, Err(Error::PathologicalRegex { index: 0 })));
    }

    /// Non-pathological repetition `a+` compiles fine.
    #[test]
    fn safe_regex_accepted() {
        let set = PatternSet::builder().regex(r"a+").build().unwrap();
        assert_eq!(set.len(), 1);
    }

    /// Nested star `(a*)*` is also pathological.
    #[test]
    fn pathological_nested_star_rejected() {
        let result = PatternSet::builder().regex(r"(a*)*").build();
        assert!(matches!(result, Err(Error::PathologicalRegex { .. })));
    }

    /// Pathological pattern in a mixed set rejects the whole build.
    #[test]
    fn pathological_in_mixed_set_rejects() {
        let result = PatternSet::builder()
            .literal("safe")
            .regex(r"([0-9]+)+")
            .build();
        assert!(matches!(result, Err(Error::PathologicalRegex { index: 1 })));
    }
}
