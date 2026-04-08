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
fn load_literal_count_zero_with_offsets() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let mut corrupted = bytes.clone();

    // Set literal count to 0 but offsets table has entries
    corrupted[20..24].copy_from_slice(&0u32.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Literal count 0 with offsets should error");
}
