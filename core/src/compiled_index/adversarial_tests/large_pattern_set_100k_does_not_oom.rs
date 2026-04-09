//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::CompiledPatternIndex;
use crate::PatternSet;

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

// =============================================================================
// TEST 9: Maximum pattern count (100K) — must not OOM during build
// =============================================================================

#[test]
fn large_pattern_set_100k_does_not_oom() {
    // Build a pattern set with 100K patterns
    let mut builder = PatternSet::builder();
    for i in 0..100_000 {
        // Generate unique patterns of varying lengths
        let pattern = format!("pattern_{:08x}", i);
        builder = builder.literal(&pattern);
    }

    let pattern_set = builder.build().expect("Should build 100K patterns");
    assert_eq!(pattern_set.len(), 100_000);

    // Build index should not OOM
    let bytes =
        CompiledPatternIndex::build(&pattern_set).expect("Should build index for 100K patterns");
    assert!(!bytes.is_empty(), "Index should not be empty");

    // Load index should not OOM
    let index = CompiledPatternIndex::load(&bytes).expect("Should load index for 100K patterns");
    assert_eq!(index.names().len(), 100_000);

    // Scan should work and find matches
    let test_data = b"pattern_00000000 pattern_0001869f pattern_000186a0";
    let matches = index
        .scan(test_data)
        .expect("Should scan with 100K patterns");
    assert!(!matches.is_empty(), "Should find matches");
}
