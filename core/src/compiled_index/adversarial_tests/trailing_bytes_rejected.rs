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
fn trailing_bytes_rejected() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let mut corrupted = bytes.clone();

    // Append garbage bytes
    corrupted.extend_from_slice(b"TRAILING_GARBAGE_BYTES");

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Trailing bytes should be rejected");

    match result {
        Err(Error::PatternCompilationFailed { reason }) => {
            assert!(
                reason.contains("trailing") || reason.contains("CRC mismatch") || reason.contains("integrity"),
                "Error should mention trailing bytes: {}",
                reason
            );
        }
        Err(other) => panic!("Wrong error type: {:?}", other),
        Ok(_) => unreachable!(),
    }
}
