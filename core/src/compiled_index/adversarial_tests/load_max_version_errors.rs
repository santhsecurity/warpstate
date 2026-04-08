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
fn load_max_version_errors() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let mut corrupted = CompiledPatternIndex::build(&patterns).unwrap();

    // u32::MAX version should error
    corrupted[8..12].copy_from_slice(&u32::MAX.to_le_bytes());

    let result = CompiledPatternIndex::load(&corrupted);
    assert!(result.is_err(), "Max version should error");
}
