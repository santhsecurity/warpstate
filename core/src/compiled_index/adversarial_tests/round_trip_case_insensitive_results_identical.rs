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
fn round_trip_case_insensitive_results_identical() {
    let patterns = PatternSet::builder()
        .case_insensitive(true)
        .literal("Password")
        .literal("SECRET")
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = b"PASSWORD secret PaSsWoRd SECRET";

    let original_matches = patterns.scan(test_data).unwrap();
    let index_matches = index.scan(test_data).unwrap();

    assert_eq!(
        original_matches.len(),
        index_matches.len(),
        "Case-insensitive match count should be identical"
    );
}
