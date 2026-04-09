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
fn negative_length_prefixed_fields() {
    // u32::MAX as a length should be handled (it's used for None names)
    // but other very large lengths should error

    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let mut corrupted = bytes.clone();

    // u32::MAX as length for packed_bytes is definitely wrong
    corrupted[32..36].copy_from_slice(&u32::MAX.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "u32::MAX packed_bytes length should error");
}
