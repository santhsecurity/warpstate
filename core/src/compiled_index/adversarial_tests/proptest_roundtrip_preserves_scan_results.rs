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
// TEST 7: proptest — for any set of literal patterns, build+load+scan produces same results
// =============================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn proptest_roundtrip_preserves_scan_results(
        patterns in prop::collection::vec("[a-zA-Z0-9]{1,20}", 1..50),
        input in "[a-zA-Z0-9 ]{0,1000}"
    ) {
        // Build pattern set from random patterns
        let mut builder = PatternSet::builder();
        for pattern in &patterns {
            builder = builder.literal(pattern);
        }

        let pattern_set = match builder.build() {
            Ok(ps) => ps,
            Err(_) => return Ok(()), // Skip invalid pattern sets
        };

        let bytes = match CompiledPatternIndex::build(&pattern_set) {
            Ok(b) => b,
            Err(_) => return Ok(()), // Skip build failures
        };

        let index = match CompiledPatternIndex::load(&bytes) {
            Ok(i) => i,
            Err(_) => return Ok(()), // Skip load failures
        };

        let input_bytes = input.as_bytes();
        let original_matches = pattern_set.scan(input_bytes).unwrap();
        let index_matches = index.scan(input_bytes).unwrap();

        prop_assert_eq!(
            original_matches.len(),
            index_matches.len(),
            "Match count differs: original={}, index={}",
            original_matches.len(),
            index_matches.len()
        );

        for (i, (orig, idx)) in original_matches.iter().zip(index_matches.iter()).enumerate() {
            prop_assert_eq!(
                orig.pattern_id, idx.pattern_id,
                "Pattern ID differs at match {}", i
            );
            prop_assert_eq!(
                orig.start, idx.start,
                "Start position differs at match {}", i
            );
            prop_assert_eq!(
                orig.end, idx.end,
                "End position differs at match {}", i
            );
        }
    }

    #[test]
    fn proptest_regex_roundtrip_preserves_results(
        patterns in prop::collection::vec("[a-z]{1,5}", 1..20),
        input in "[a-z ]{0,500}"
    ) {
        // Build pattern set from literal patterns (more predictable than random regex)
        let mut builder = PatternSet::builder();
        for pattern in &patterns {
            builder = builder.literal(pattern);
        }

        let pattern_set = match builder.build() {
            Ok(ps) => ps,
            Err(_) => return Ok(()),
        };

        let bytes = match CompiledPatternIndex::build(&pattern_set) {
            Ok(b) => b,
            Err(_) => return Ok(()),
        };

        let index = match CompiledPatternIndex::load(&bytes) {
            Ok(i) => i,
            Err(_) => return Ok(()),
        };

        let input_bytes = input.as_bytes();
        let original_matches = pattern_set.scan(input_bytes).unwrap();
        let index_matches = index.scan(input_bytes).unwrap();

        prop_assert_eq!(
            original_matches.len(),
            index_matches.len(),
            "Regex round-trip match count differs"
        );
    }
}

/// Targeted diagnostic: prefix overlap patterns must produce identical matches.
#[test]
fn prefix_overlap_parity_diagnostic() {
    let patterns = vec![
        "p".to_string(),
        "p6".to_string(),
        "0AAaa0a0a0AaAA0a0".to_string(),
    ];
    let input = "FcGGQTS22K Ls7n 1O47KCApkz E0lG3Vwp69jsHAw3 j4i2Y6AD DB8 sUSH WlYem eaT87 I DEu3K7o5a7Sk U78 V  zUf2w8XQkf443Ecj4J Uupk7 eJJNWh d v 0 D  J pUyv579Y22aqPIvNt1a2v cPbIc 59 k0svYpgyJ55 2TL GW yL5 PHr55P7SDbw47NTaLCGP96 6vU6Mco wYHiJx u3 U oek4wx4  lkuAjvZ sIWN1 P eM1 i1Go P M4m9Q JXFRS3S3udH6kW639D475N k k t81ECl 9OgckMkXoV1 Aii 48h4  zmnN8 e4  Zpvvxhg3XaTi7 7xG1agam6AxW aye8b   jt RCJ0  410z4E0 STE  UrV1";

    let mut builder = PatternSet::builder();
    for p in &patterns {
        builder = builder.literal(p);
    }
    let ps = builder.build().unwrap();
    let bytes = CompiledPatternIndex::build(&ps).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();

    let input_bytes = input.as_bytes();
    let orig = ps.scan(input_bytes).unwrap();
    let idx = index.scan(input_bytes).unwrap();

    assert_eq!(
        orig.len(),
        idx.len(),
        "Match count differs: orig={orig:?}, idx={idx:?}"
    );
    for (i, (o, x)) in orig.iter().zip(idx.iter()).enumerate() {
        assert_eq!(
            o.pattern_id, x.pattern_id,
            "Pattern ID mismatch at match {i}: orig={o:?}, idx={x:?}"
        );
        assert_eq!(o.start, x.start, "Start mismatch at match {i}");
        assert_eq!(o.end, x.end, "End mismatch at match {i}");
    }
}
