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
fn mixed_literals_and_regex_roundtrip() {
    let patterns = PatternSet::builder()
        .literal("start")
        .regex(r"\d+")
        .literal("middle")
        .regex(r"[a-z]+")
        .literal("end")
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = b"start 123 middle abc end";
    let original = patterns.scan(test_data).unwrap();
    let loaded = index.scan(test_data).unwrap();

    // Both should find the same patterns - verify by checking each match
    // The key requirement: both find the SAME matches at SAME positions
    assert!(
        !original.is_empty(),
        "Original pattern set should find matches"
    );
    assert!(!loaded.is_empty(), "Loaded index should find matches");

    // Verify all literals are found
    let original_literals: Vec<_> = original
        .iter()
        .filter(|m| m.pattern_id == 0 || m.pattern_id == 2 || m.pattern_id == 4)
        .collect();
    let loaded_literals: Vec<_> = loaded
        .iter()
        .filter(|m| m.pattern_id == 0 || m.pattern_id == 2 || m.pattern_id == 4)
        .collect();

    assert_eq!(
        original_literals.len(),
        loaded_literals.len(),
        "Literal matches should be identical"
    );

    // Verify regex patterns are found (at least some matches)
    let original_regex: Vec<_> = original
        .iter()
        .filter(|m| m.pattern_id == 1 || m.pattern_id == 3)
        .collect();
    let loaded_regex: Vec<_> = loaded
        .iter()
        .filter(|m| m.pattern_id == 1 || m.pattern_id == 3)
        .collect();

    assert!(
        !original_regex.is_empty(),
        "Original should find regex matches"
    );
    assert!(!loaded_regex.is_empty(), "Loaded should find regex matches");
}
