//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::CompiledPatternIndex;
use crate::PatternSet;

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

#[test]
fn overlapping_patterns_preserved() {
    let patterns = PatternSet::builder()
        .literal("abc")
        .literal("bcd")
        .literal("cde")
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    // Non-overlapping scan - first match at each position wins
    let test_data = b"abcde";
    let original = patterns.scan(test_data).unwrap();
    let loaded = index.scan(test_data).unwrap();

    assert_eq!(original.len(), loaded.len());
}
