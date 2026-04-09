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
// TEST 4: Literal count != actual literals — must detect
// =============================================================================

#[test]
fn load_literal_count_mismatch_detected() {
    let patterns = PatternSet::builder()
        .literal("first")
        .literal("second")
        .literal("third")
        .build()
        .unwrap();

    let mut corrupted = CompiledPatternIndex::build(&patterns).unwrap();

    // Find and corrupt the literal count field (at offset 20: after magic(8) + version(4) + flags(4) + pattern_count(4))
    let wrong_count = 5u32; // Claim 5 literals but only have 3
    corrupted[20..24].copy_from_slice(&wrong_count.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Literal count mismatch should error");

    match result {
        Err(Error::PatternCompilationFailed { reason }) => {
            assert!(
                (reason.contains("literal") && reason.contains("inconsistent"))
                    || reason.contains("CRC mismatch")
                    || reason.contains("integrity"),
                "Error should mention literal inconsistency or CRC corruption: {}",
                reason
            );
        }
        Err(other) => panic!("Wrong error type: {:?}", other),
        Ok(_) => unreachable!(),
    }
}
