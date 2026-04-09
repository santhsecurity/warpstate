//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::CompiledPatternIndex;
use crate::PatternSet;

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

#[test]
fn very_long_literal_pattern() {
    // Pattern at the edge of reasonable size
    let long_pattern = "a".repeat(1000);

    let patterns = PatternSet::builder()
        .literal(&long_pattern)
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let test_data = format!("prefix{}suffix", long_pattern);
    let matches = index.scan(test_data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].end - matches[0].start, 1000);
}
