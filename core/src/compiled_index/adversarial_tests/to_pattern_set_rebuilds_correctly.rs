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
fn to_pattern_set_rebuilds_correctly() {
    let patterns = PatternSet::builder()
        .literal("alpha")
        .regex(r"b[0-9]+")
        .named_literal("key", "password")
        .build()
        .unwrap();

    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    // Rebuild the pattern set
    let rebuilt = index.to_pattern_set().unwrap();

    // Both should produce identical scan results
    let test_data = b"alpha b42 password beta b123";
    let original = patterns.scan(test_data).unwrap();
    let from_rebuilt = rebuilt.scan(test_data).unwrap();

    assert_eq!(original.len(), from_rebuilt.len());
    for (o, r) in original.iter().zip(from_rebuilt.iter()) {
        assert_eq!(o.pattern_id, r.pattern_id);
        assert_eq!(o.start, r.start);
        assert_eq!(o.end, r.end);
    }
}
