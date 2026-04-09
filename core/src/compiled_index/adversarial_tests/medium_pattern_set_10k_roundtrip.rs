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
fn medium_pattern_set_10k_roundtrip() {
    let mut builder = PatternSet::builder();
    for i in 0..10_000 {
        let pattern = format!("p{:04x}", i);
        builder = builder.literal(&pattern);
    }

    let pattern_set = builder.build().expect("Should build 10K patterns");
    let bytes = CompiledPatternIndex::build(&pattern_set).expect("Should build index");
    let index = CompiledPatternIndex::load(&bytes).expect("Should load index");

    // Verify round-trip scanning works correctly
    let test_data = b"p0000 p1234 p270f p2710 pffff";
    let original_matches = pattern_set.scan(test_data).unwrap();
    let index_matches = index.scan(test_data).unwrap();

    assert_eq!(original_matches.len(), index_matches.len());
}
