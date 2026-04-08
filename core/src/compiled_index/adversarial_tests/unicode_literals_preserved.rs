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
fn unicode_literals_preserved() {
    let patterns = PatternSet::builder()
        .literal("日本語")
        .literal("🎉emoji🎊")
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = "日本語 test 🎉emoji🎊".as_bytes();
    let original = patterns.scan(test_data).unwrap();
    let loaded = index.scan(test_data).unwrap();

    assert_eq!(original.len(), loaded.len());
}
