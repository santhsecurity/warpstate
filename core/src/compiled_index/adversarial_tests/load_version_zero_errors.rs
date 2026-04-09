//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::CompiledPatternIndex;
use crate::Error;
use crate::PatternSet;

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

#[test]
fn load_version_zero_errors() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let mut corrupted = CompiledPatternIndex::build(&patterns).unwrap();

    // Version 0 is explicitly rejected
    corrupted[8..12].copy_from_slice(&0u32.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Version 0 should error");

    match result {
        Err(Error::PatternCompilationFailed { reason }) => {
            assert!(
                reason.contains("version") && reason.contains("invalid"),
                "Error should mention invalid version: {}",
                reason
            );
        }
        Err(other) => panic!("Wrong error type: {:?}", other),
        Ok(_) => unreachable!(),
    }
}
