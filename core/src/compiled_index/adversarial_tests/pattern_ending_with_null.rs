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

#[test]
fn pattern_ending_with_null() {
    let pattern_set = PatternSet::builder()
        .literal_bytes(vec![0x41, 0x42, 0x00]) // AB\0
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&pattern_set).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = b"\x41\x42\x00";
    let matches = index.scan(test_data).unwrap();
    assert_eq!(matches.len(), 1, "Should find pattern ending with null");
}
