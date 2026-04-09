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
// TEST 8: Empty pattern set → build → load — must work correctly
// =============================================================================

#[test]
fn empty_pattern_set_errors_on_build() {
    let result = PatternSet::builder().build();
    assert!(
        matches!(result, Err(Error::EmptyPatternSet)),
        "Empty pattern set should error"
    );
}
