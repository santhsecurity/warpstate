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
fn oversized_packed_bytes_length_detected() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let mut corrupted = bytes.clone();

    // Find the packed_bytes length field (comes after header, at offset 32)
    // It's a u32 length prefix
    let packed_len_offset = 32;

    // Set length to more than available data
    corrupted[packed_len_offset..packed_len_offset + 4]
        .copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(
        result.is_err(),
        "Oversized packed_bytes length should error"
    );
}
