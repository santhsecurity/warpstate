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
// ADDITIONAL ADVERSARIAL TESTS — Edge cases that could corrupt the index
// =============================================================================

#[test]
fn pattern_count_mismatch_detected() {
    let patterns = PatternSet::builder()
        .literal("a")
        .literal("b")
        .literal("c")
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let mut corrupted = bytes.clone();

    // Corrupt pattern count to be higher than actual names
    // Pattern count is at offset 16: after magic(8) + version(4) + flags(4)
    corrupted[16..20].copy_from_slice(&100u32.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Pattern count mismatch should error");
}
