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
fn load_wrong_magic_bytes_errors() {
    // Start with valid index data
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let valid_bytes = CompiledPatternIndex::build(&patterns).unwrap();

    // Wrong magic bytes - various corrupted forms (all must error, not panic)
    // We pad with 0xFF to ensure we don't accidentally create valid partial data
    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        ("garbage", b"NOTINDEX".to_vec()),
        ("wrong_version_magic", b"WPSIDX00".to_vec()),
        ("wrong_version_magic2", b"WPSIDX02".to_vec()),
        ("case_wrong", b"wpsidx01".to_vec()),
        ("random_bytes", vec![0xFF; 8]),
        ("zeros", vec![0x00; 8]),
        ("null_terminated", b"WPSIDX01\0".to_vec()),
        (
            "partial_magic",
            vec![b'W', b'P', b'S', 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        ),
    ];

    for (name, wrong_magic) in test_cases {
        let mut corrupted = wrong_magic.clone();
        // Pad with garbage to reach valid_bytes length (but don't create valid magic)
        if corrupted.len() < valid_bytes.len() {
            corrupted.extend(vec![0xFF; valid_bytes.len() - corrupted.len()]);
        }

        let result = CompiledPatternIndex::load(&corrupted);
        assert!(
            result.is_err(),
            "Test case '{}' should error with wrong magic",
            name
        );

        // Ensure it errors with PatternCompilationFailed - the key is it doesn't panic
        match result {
            Err(Error::PatternCompilationFailed { .. }) => {
                // Any PatternCompilationFailed error is acceptable
            }
            Err(_) => {
                // Any error is acceptable - the key is it doesn't panic
            }
            Ok(_) => panic!("Test case '{}' should have errored", name),
        }
    }

    // Empty data should also error (not panic)
    let result = CompiledPatternIndex::load(b"");
    assert!(result.is_err(), "Empty data should error");
}
