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
// TEST 2: Version > current — must error
// =============================================================================

#[test]
fn load_future_version_errors() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let mut corrupted = CompiledPatternIndex::build(&patterns).unwrap();

    // Corrupt version to a future version
    let future_version = VERSION + 1;
    corrupted[8..12].copy_from_slice(&future_version.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Future version should error");

    match result {
        Err(Error::PatternCompilationFailed { reason }) => {
            assert!(
                reason.contains("version") && reason.contains("unsupported"),
                "Error should mention unsupported version: {}",
                reason
            );
        }
        Err(other) => panic!("Wrong error type: {:?}", other),
        Ok(_) => unreachable!(),
    }
}
