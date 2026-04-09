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
fn pattern_starting_with_null() {
    let pattern_set = PatternSet::builder()
        .literal_bytes(vec![0x00, 0x41, 0x42]) // \0AB
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&pattern_set).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = b"\x00\x41\x42";
    let matches = index.scan(test_data).unwrap();
    assert_eq!(matches.len(), 1, "Should find pattern starting with null");
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 3);
}
