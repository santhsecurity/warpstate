//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::{CompiledPatternIndex, MAGIC, VERSION};
use crate::Error;
use crate::PatternSet;

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

// =============================================================================
// TEST 10: Pattern with null bytes — must round-trip correctly
// =============================================================================

#[test]
fn null_bytes_pattern_roundtrip() {
    // Pattern containing null bytes
    let pattern_with_null = b"prefix\x00\x00\x00suffix";

    let pattern_set = PatternSet::builder()
        .literal_bytes(pattern_with_null.to_vec())
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&pattern_set).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    // Scan data containing the null-byte pattern
    let test_data = b"prefix\x00\x00\x00suffix";
    let original_matches = pattern_set.scan(test_data).unwrap();
    let index_matches = index.scan(test_data).unwrap();

    assert_eq!(
        original_matches.len(),
        index_matches.len(),
        "Match count should be identical for null-byte patterns"
    );

    // Verify the pattern bytes are preserved through round-trip
    let rebuilt = index.to_pattern_set().unwrap();
    let rebuilt_matches = rebuilt.scan(test_data).unwrap();
    assert_eq!(
        original_matches.len(),
        rebuilt_matches.len(),
        "Rebuilt pattern set should produce same matches"
    );
}
