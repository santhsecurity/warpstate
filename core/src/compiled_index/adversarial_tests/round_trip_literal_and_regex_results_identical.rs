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
// TEST 6: Round-trip: build → save → load → scan — results identical
// =============================================================================

#[test]
fn round_trip_literal_and_regex_results_identical() {
    let patterns = PatternSet::builder()
        .literal("password")
        .regex(r"[A-Z]{3}-\d+")
        .literal("secret_key")
        .named_regex("api_key", r"sk-[a-zA-Z0-9]{32}")
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = b"password: sk-abc123xyz789ABCDEF1234567890, code: ABC-123, secret_key here";

    let original_matches = patterns.scan(test_data).unwrap();
    let index_matches = index.scan(test_data).unwrap();

    assert_eq!(
        original_matches.len(),
        index_matches.len(),
        "Match count should be identical"
    );

    for (orig, idx) in original_matches.iter().zip(index_matches.iter()) {
        assert_eq!(orig.pattern_id, idx.pattern_id, "Pattern IDs should match");
        assert_eq!(orig.start, idx.start, "Start positions should match");
        assert_eq!(orig.end, idx.end, "End positions should match");
    }
}
