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
// TEST 3: Truncated data — must error at exact boundary
// =============================================================================

#[test]
fn load_truncated_at_various_boundaries() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let valid_bytes = CompiledPatternIndex::build(&patterns).unwrap();

    // Test truncation at every byte boundary
    for truncate_at in 1..valid_bytes.len() {
        let truncated = &valid_bytes[..truncate_at];
        let result = CompiledPatternIndex::load(truncated);

        assert!(
            result.is_err(),
            "Truncation at byte {} should error",
            truncate_at
        );
    }
}
