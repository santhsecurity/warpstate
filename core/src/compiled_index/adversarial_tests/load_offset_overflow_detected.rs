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
fn load_offset_overflow_detected() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let layout = &index.layout;

    let mut corrupted = bytes.clone();
    let offsets_start = layout.offsets_range.start;

    // Find the first offset and set it to cause overflow when start + len is computed
    let first_offset_pos = offsets_start + 4; // After count

    // Set start to max u32 and len to 1 - will overflow
    corrupted[first_offset_pos..first_offset_pos + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    corrupted[first_offset_pos + 4..first_offset_pos + 8].copy_from_slice(&1u32.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Offset overflow should error");
}
