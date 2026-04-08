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
fn single_literal_optimization_preserved() {
    // Single literal uses optimized memmem path
    let patterns = PatternSet::builder().literal("needle").build().unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = b"haystack needle haystack needle";
    let original = patterns.scan(test_data).unwrap();
    let loaded = index.scan(test_data).unwrap();

    assert_eq!(original.len(), 2);
    assert_eq!(loaded.len(), 2);
    assert_eq!(original[0].start, loaded[0].start);
}
