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
fn load_truncated_header_fields() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let valid_bytes = CompiledPatternIndex::build(&patterns).unwrap();

    // Header is: magic(8) + version(4) + flags(4) + pattern_count(4) + literal_count(4) + regex_count(4) + hash_window(4) = 32 bytes
    let header_size = 8 + 4 + 4 + 4 + 4 + 4 + 4;

    // Test truncation within header
    for truncate_at in [8, 9, 12, 16, 20, 24, 28, 31] {
        let truncated = &valid_bytes[..truncate_at];
        let result = CompiledPatternIndex::load(truncated);

        assert!(
            result.is_err(),
            "Truncation at header byte {} should error",
            truncate_at
        );

        match result {
            Err(Error::PatternCompilationFailed { reason }) => {
                assert!(
                    reason.contains("truncated")
                        || reason.contains("CRC mismatch")
                        || reason.contains("integrity")
                        || reason.contains("overflowed"),
                    "Error should mention truncation: {}",
                    reason
                );
            }
            Err(other) => panic!("Wrong error type at {}: {:?}", truncate_at, other),
            Ok(_) => unreachable!(),
        }
    }
}
