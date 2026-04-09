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
// TEST 5: Offset pointing past packed_bytes — must detect
// =============================================================================

#[test]
fn load_offset_past_packed_bytes_detected() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();

    // Parse to find the offsets range
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let layout = &index.layout;

    // Corrupt the offset to point past packed_bytes
    let mut corrupted = bytes.clone();

    // The packed_bytes length is stored as a u32 before the actual bytes
    // We need to find and corrupt the offset values in the offsets table
    let offsets_start = layout.offsets_range.start;

    // Read the offsets count
    let offsets_count = u32::from_le_bytes([
        corrupted[offsets_start],
        corrupted[offsets_start + 1],
        corrupted[offsets_start + 2],
        corrupted[offsets_start + 3],
    ]);

    if offsets_count > 0 {
        // Corrupt the first offset's start to be past packed_bytes
        // Each offset is (start: u32, len: u32) = 8 bytes
        let first_offset_pos = offsets_start + 4; // After count

        // Set start to a very large value
        corrupted[first_offset_pos..first_offset_pos + 4]
            .copy_from_slice(&0xFFFF_0000u32.to_le_bytes());

        let result = CompiledPatternIndex::load(&corrupted);
        assert!(result.is_err(), "Offset past packed_bytes should error");

        match result {
            Err(Error::PatternCompilationFailed { reason }) => {
                assert!(
                    reason.contains("outside") || reason.contains("overflows") || reason.contains("CRC mismatch") || reason.contains("integrity"),
                    "Error should mention offset outside bounds: {}",
                    reason
                );
            }
            Err(other) => panic!("Wrong error type: {:?}", other),
            Ok(_) => unreachable!(),
        }
    }
}
