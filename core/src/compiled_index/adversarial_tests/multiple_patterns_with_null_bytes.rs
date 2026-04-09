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
fn multiple_patterns_with_null_bytes() {
    let patterns = vec![
        vec![0x00],
        vec![0x00, 0x00],
        vec![0xFF, 0x00, 0xFF],
        b"hello\x00world".to_vec(),
        b"\x00\x01\x02\x03".to_vec(),
    ];

    let mut builder = PatternSet::builder();
    for pattern in &patterns {
        builder = builder.literal_bytes(pattern.clone());
    }

    let pattern_set = builder.build().unwrap();
    let bytes = CompiledPatternIndex::build(&pattern_set).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    // Test data containing all patterns
    let test_data = b"\x00\x00\x00hello\x00world\x00\x01\x02\x03";
    let original_matches = pattern_set.scan(test_data).unwrap();
    let index_matches = index.scan(test_data).unwrap();

    assert_eq!(
        original_matches.len(),
        index_matches.len(),
        "Multiple null-byte patterns should round-trip correctly"
    );
}
