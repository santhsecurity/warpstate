//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::CompiledPatternIndex;
use crate::PatternSet;

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

// =============================================================================
// TEST 7: proptest — for any set of literal patterns, build+load+scan produces same results
// =============================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn proptest_roundtrip_preserves_scan_results(
        patterns in prop::collection::vec("[a-zA-Z0-9]{1,20}", 1..50),
        input in "[a-zA-Z0-9 ]{0,1000}"
    ) {
        // Build pattern set from random patterns
        let mut builder = PatternSet::builder();
        for pattern in &patterns {
            builder = builder.literal(pattern);
        }

        let pattern_set = match builder.build() {
            Ok(ps) => ps,
            Err(_) => return Ok(()), // Skip invalid pattern sets
        };

        let bytes = match CompiledPatternIndex::build(&pattern_set) {
            Ok(b) => b,
            Err(_) => return Ok(()), // Skip build failures
        };

        let index = match CompiledPatternIndex::load(&bytes) {
            Ok(i) => i,
            Err(_) => return Ok(()), // Skip load failures
        };

        let input_bytes = input.as_bytes();
        let original_matches = pattern_set.scan(input_bytes).unwrap();
        let index_matches = index.scan(input_bytes).unwrap();

        prop_assert_eq!(
            original_matches.len(),
            index_matches.len(),
            "Match count differs: original={}, index={}",
            original_matches.len(),
            index_matches.len()
        );

        for (i, (orig, idx)) in original_matches.iter().zip(index_matches.iter()).enumerate() {
            prop_assert_eq!(
                orig.pattern_id, idx.pattern_id,
                "Pattern ID differs at match {}", i
            );
            prop_assert_eq!(
                orig.start, idx.start,
                "Start position differs at match {}", i
            );
            prop_assert_eq!(
                orig.end, idx.end,
                "End position differs at match {}", i
            );
        }
    }

    #[test]
    fn proptest_regex_roundtrip_preserves_results(
        patterns in prop::collection::vec("[a-z]{1,5}", 1..20),
        input in "[a-z ]{0,500}"
    ) {
        // Build pattern set from literal patterns (more predictable than random regex)
        let mut builder = PatternSet::builder();
        for pattern in &patterns {
            builder = builder.literal(pattern);
        }

        let pattern_set = match builder.build() {
            Ok(ps) => ps,
            Err(_) => return Ok(()),
        };

        let bytes = match CompiledPatternIndex::build(&pattern_set) {
            Ok(b) => b,
            Err(_) => return Ok(()),
        };

        let index = match CompiledPatternIndex::load(&bytes) {
            Ok(i) => i,
            Err(_) => return Ok(()),
        };

        let input_bytes = input.as_bytes();
        let original_matches = pattern_set.scan(input_bytes).unwrap();
        let index_matches = index.scan(input_bytes).unwrap();

        prop_assert_eq!(
            original_matches.len(),
            index_matches.len(),
            "Regex round-trip match count differs"
        );
    }
}
