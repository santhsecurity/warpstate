//! Adversarial tests for warpstate PatternSet - Core Pattern Matching Engine
//!
//! These tests are designed to FAIL and expose bugs in the pattern matching engine.
//! warpscan scans the ENTIRE internet's software supply chain.
//! A single false negative means malware goes undetected.
//!
//! CORE LAWS:
//! 1. NO STUBS — every test is fully implemented
//! 2. MODULAR — single responsibility per test
//! 3. TEST EVERYTHING — adversarial input, edge cases, crash recovery
//! 4. EVERY FINDING IS CRITICAL — at internet scale, a "low" bug corrupts billions of records
//! 5. Actionable errors, no dead code, no swallowed errors

use proptest::collection::vec;
use proptest::prelude::*;
use warpstate::{Error, Match, PatternSet};

// =============================================================================
// Test 1: Pattern at byte 0 of input - must match
// =============================================================================

/// Pattern at the absolute start of input must be detected.
/// Fix: If this fails, the scanner has an off-by-one error at the start boundary.
#[test]
fn pattern_at_byte_zero_must_match() {
    let ps = PatternSet::builder().literal("START").build().unwrap();

    let matches = ps.scan(b"START of input").unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Pattern at byte 0 not found. Scanner has start boundary bug."
    );
    assert_eq!(
        matches[0].start, 0,
        "CRITICAL: Match start position should be 0, got {}",
        matches[0].start
    );
    assert_eq!(matches[0].end, 5);
}

/// Single byte pattern at position 0.
#[test]
fn single_byte_pattern_at_byte_zero() {
    let ps = PatternSet::builder().literal("X").build().unwrap();

    let matches = ps.scan(b"X").unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Single byte pattern at position 0 not found."
    );
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 1);
}

// =============================================================================
// Test 2: Pattern at last byte of input - must match
// =============================================================================

/// Pattern ending at the exact last byte must be detected.
/// Fix: If this fails, the scanner has an off-by-one error at the end boundary.
#[test]
fn pattern_at_last_byte_must_match() {
    let ps = PatternSet::builder().literal("END").build().unwrap();

    let input = b"input ends with END";
    let matches = ps.scan(input).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Pattern at end not found. Scanner has end boundary bug."
    );
    assert_eq!(
        matches[0].end as usize,
        input.len(),
        "CRITICAL: Match should end at input length {}, ended at {}",
        input.len(),
        matches[0].end
    );
    assert_eq!(matches[0].start, 16);
}

/// Pattern consuming entire input (start=0, end=len).
#[test]
fn pattern_spanning_entire_input() {
    let ps = PatternSet::builder().literal("exact").build().unwrap();

    let matches = ps.scan(b"exact").unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Pattern spanning entire input not found."
    );
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 5);
}

// =============================================================================
// Test 3: Pattern spanning exact end of input (partial) - must NOT match
// =============================================================================

/// Partial pattern at end must NOT match - this is a critical security test.
/// A partial match could bypass detection of malicious patterns.
/// Fix: If this matches, the scanner has a buffer over-read vulnerability.
#[test]
fn partial_pattern_at_end_must_not_match() {
    let ps = PatternSet::builder().literal("hello").build().unwrap();

    // Input ends with "hel" - only partial match for "hello"
    let matches = ps.scan(b"prefix hel").unwrap();

    assert_eq!(
        matches.len(),
        0,
        "CRITICAL SECURITY BUG: Partial pattern matched at end! \"hel\" should NOT match \"hello\""
    );
}

/// Partial match with just one byte missing.
#[test]
fn partial_pattern_one_byte_short() {
    let ps = PatternSet::builder().literal("abcd").build().unwrap();

    // "abc" is one byte short of "abcd"
    let matches = ps.scan(b"prefix abc").unwrap();

    assert_eq!(
        matches.len(),
        0,
        "CRITICAL: Pattern \"abc\" (3 bytes) should NOT match \"abcd\" (4 bytes)"
    );
}

/// Pattern longer than input should never match.
#[test]
fn pattern_longer_than_input_must_not_match() {
    let ps = PatternSet::builder()
        .literal("longpattern")
        .build()
        .unwrap();

    let matches = ps.scan(b"short").unwrap();

    assert_eq!(
        matches.len(),
        0,
        "CRITICAL: Pattern longer than input should never match"
    );
}

/// Verify partial match at every possible offset near end.
#[test]
fn partial_match_all_offsets_near_end() {
    let pattern = "TARGET";
    let ps = PatternSet::builder().literal(pattern).build().unwrap();

    // Test with input ending in "T", "TA", "TAR", "TARG", "TARGE"
    for len in 1..pattern.len() {
        let partial = &pattern[..len];
        let input = format!("prefix{}", partial);
        let matches = ps.scan(input.as_bytes()).unwrap();

        assert_eq!(
            matches.len(), 0,
            "CRITICAL: Partial pattern \"{}\" ({} bytes) should NOT match full pattern \"{}\" ({} bytes)",
            partial, len, pattern, pattern.len()
        );
    }
}

// =============================================================================
// Test 4: Overlapping patterns - both must be found
// =============================================================================

/// Multiple patterns at the same position must all be reported.
/// Fix: If any pattern is missing, the scanner has overlapping match bugs.
#[test]
fn overlapping_patterns_both_must_be_found() {
    let ps = PatternSet::builder()
        .literal("abc")
        .literal("bcd")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abcd").unwrap();

    let abc_found = matches.iter().any(|m| m.pattern_id == 0);
    let bcd_found = matches.iter().any(|m| m.pattern_id == 1);

    assert!(
        abc_found,
        "CRITICAL: Pattern 'abc' not found in overlapping scan. Pattern ID 0 missing."
    );
    assert!(
        bcd_found,
        "CRITICAL: Pattern 'bcd' not found in overlapping scan. Pattern ID 1 missing."
    );
}

/// Identical patterns with different IDs must both match.
#[test]
fn identical_patterns_different_ids_both_found() {
    let ps = PatternSet::builder()
        .literal("SAME")
        .literal("SAME")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"SAME").unwrap();

    assert_eq!(
        matches.len(),
        2,
        "CRITICAL: Two identical patterns should produce 2 matches, got {}",
        matches.len()
    );

    let pattern_ids: Vec<_> = matches.iter().map(|m| m.pattern_id).collect();
    assert_ne!(
        pattern_ids[0], pattern_ids[1],
        "CRITICAL: Identical patterns should have different pattern IDs"
    );
}

/// Overlapping with different lengths at same start position.
#[test]
fn overlapping_different_lengths_same_start() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("ab")
        .literal("abc")
        .literal("abcd")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abcde").unwrap();

    // All four patterns should match starting at position 0
    let start_at_0: Vec<_> = matches.iter().filter(|m| m.start == 0).collect();

    assert_eq!(
        start_at_0.len(),
        4,
        "CRITICAL: Expected 4 patterns matching at position 0, found {}",
        start_at_0.len()
    );
}

// =============================================================================
// Test 5: 10K matches in one file - all must be reported (no silent overflow)
// =============================================================================

/// EXACTLY 10,000 matches must be reported without silent truncation.
/// This tests for buffer overflow handling.
/// Fix: If matches are silently dropped, the engine has a critical security vulnerability.
#[test]
fn ten_thousand_matches_all_reported() {
    let ps = PatternSet::builder().literal("X").build().unwrap();

    // Create input that produces exactly 10,000 matches
    let input = vec![b'X'; 10_000];

    let matches = ps.scan(&input).unwrap();

    assert_eq!(
        matches.len(),
        10_000,
        "CRITICAL: Expected 10,000 matches, got {}. Silent buffer overflow detected!",
        matches.len()
    );

    // Verify all positions are correct
    for (i, m) in matches.iter().enumerate() {
        assert_eq!(
            m.start as usize, i,
            "CRITICAL: Match at index {} has wrong start position: expected {}, got {}",
            i, i, m.start
        );
    }
}

/// Verify no silent truncation with larger match counts.
#[test]
fn large_match_count_no_silent_truncation() {
    let ps = PatternSet::builder().literal("ab").build().unwrap();

    // "ababab..." with 5000 repetitions
    let mut input = Vec::with_capacity(10_000);
    for _ in 0..5000 {
        input.push(b'a');
        input.push(b'b');
    }

    let matches = ps.scan_overlapping(&input).unwrap();

    // "ab" pattern in "abab..." finds matches at positions 0, 2, 4, ... 9998
    // since pattern is 2 bytes and input is 10_000 bytes, we get 5000 matches
    let expected_count = 5000;

    assert_eq!(
        matches.len(),
        expected_count,
        "CRITICAL: Overlapping scan should find {} matches, got {}",
        expected_count,
        matches.len()
    );

    // Verify positions are correct (every even position)
    for (i, m) in matches.iter().enumerate() {
        let expected_start = (i * 2) as u32;
        assert_eq!(
            m.start, expected_start,
            "CRITICAL: Match {} should start at {}, started at {}",
            i, expected_start, m.start
        );
    }
}

/// Test that MatchBufferOverflow error is properly reported, not silent.
#[test]
fn match_overflow_reported_not_silent() {
    let ps = PatternSet::builder().literal("a").build().unwrap();

    // Create input that would produce more than MAX_CPU_MATCHES matches
    // MAX_CPU_MATCHES = 1_048_576
    let input = vec![b'a'; 1_048_577];

    let result = ps.scan(&input);

    match result {
        Err(Error::MatchBufferOverflow { count, max }) => {
            // Overflow correctly reported
            assert!(
                count >= max,
                "Overflow count ({}) should be >= max ({})",
                count,
                max
            );
        }
        Ok(matches) => {
            panic!(
                "CRITICAL: Expected MatchBufferOverflow error, but got {} matches silently. \
                This is a security vulnerability - overflows must be reported!",
                matches.len()
            );
        }
        Err(other) => {
            panic!("CRITICAL: Unexpected error type: {:?}", other);
        }
    }
}

// =============================================================================
// Test 6: Empty pattern - must not crash
// =============================================================================

/// Empty pattern must be rejected at build time, not crash.
#[test]
fn empty_pattern_rejected_not_crash() {
    let result = PatternSet::builder().literal("").build();

    match result {
        Err(Error::EmptyPattern { index: 0 }) => {
            // Correct behavior - empty pattern rejected
        }
        Ok(_) => {
            panic!("CRITICAL: Empty pattern was accepted - this could cause crashes during scan");
        }
        Err(other) => {
            panic!("CRITICAL: Wrong error for empty pattern: {:?}", other);
        }
    }
}

/// Empty bytes pattern must not crash.
#[test]
fn empty_bytes_pattern_rejected() {
    let result = PatternSet::builder().literal_bytes(b"").build();

    assert!(
        result.is_err(),
        "CRITICAL: Empty bytes pattern should be rejected"
    );
}

/// Empty pattern in middle of set must not crash.
#[test]
fn empty_pattern_in_middle_rejected() {
    let result = PatternSet::builder()
        .literal("first")
        .literal("")
        .literal("third")
        .build();

    match result {
        Err(Error::EmptyPattern { index: 1 }) => {
            // Correct - empty pattern at index 1
        }
        Ok(_) => {
            panic!("CRITICAL: Empty pattern in middle was accepted");
        }
        Err(other) => {
            panic!(
                "CRITICAL: Wrong error for empty pattern in middle: {:?}",
                other
            );
        }
    }
}

// =============================================================================
// Test 7: Empty input - must not crash
// =============================================================================

/// Empty input must return empty results, not crash.
#[test]
fn empty_input_returns_empty_not_crash() {
    let ps = PatternSet::builder().literal("test").build().unwrap();

    let matches = ps.scan(b"").unwrap();

    assert_eq!(
        matches.len(),
        0,
        "CRITICAL: Empty input should return 0 matches"
    );
}

/// Empty input with multiple patterns.
#[test]
fn empty_input_multiple_patterns() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("b")
        .literal("c")
        .build()
        .unwrap();

    let matches = ps.scan(b"").unwrap();

    assert!(
        matches.is_empty(),
        "CRITICAL: Empty input should produce no matches"
    );
}

/// Empty input with regex patterns.
#[test]
fn empty_input_with_regex() {
    let ps = PatternSet::builder()
        .literal("test")
        .regex(r"[a-z]+")
        .build()
        .unwrap();

    let matches = ps.scan(b"").unwrap();

    assert!(
        matches.is_empty(),
        "CRITICAL: Empty input with regex should produce no matches"
    );
}

// =============================================================================
// Test 8: Pattern longer than input - must not match, must not crash
// =============================================================================

/// Pattern longer than input must not match and must not crash.
#[test]
fn pattern_longer_than_input_no_match_no_crash() {
    let ps = PatternSet::builder()
        .literal("this is a very long pattern")
        .build()
        .unwrap();

    // Pattern (27 bytes) > input (5 bytes)
    let matches = ps.scan(b"short").unwrap();

    assert_eq!(
        matches.len(),
        0,
        "CRITICAL: Pattern longer than input should not match"
    );
}

/// Pattern exactly one byte longer than input.
#[test]
fn pattern_one_byte_longer() {
    let ps = PatternSet::builder().literal("abcd").build().unwrap();

    let matches = ps.scan(b"abc").unwrap();

    assert_eq!(
        matches.len(),
        0,
        "CRITICAL: Pattern 1 byte longer should not match"
    );
}

/// Multiple patterns, some longer, some shorter.
#[test]
fn mixed_pattern_lengths_some_longer() {
    let ps = PatternSet::builder()
        .literal("tiny")
        .literal("this_is_definitely_longer_than_input")
        .literal("small")
        .build()
        .unwrap();

    let matches = ps.scan(b"tiny small").unwrap();

    assert_eq!(
        matches.len(),
        2,
        "CRITICAL: Should match 2 patterns, found {} matches",
        matches.len()
    );

    let tiny_found = matches.iter().any(|m| m.pattern_id == 0);
    let long_found = matches.iter().any(|m| m.pattern_id == 1);
    let small_found = matches.iter().any(|m| m.pattern_id == 2);

    assert!(tiny_found, "CRITICAL: 'tiny' pattern should match");
    assert!(!long_found, "CRITICAL: Long pattern should NOT match");
    assert!(small_found, "CRITICAL: 'small' pattern should match");
}

// =============================================================================
// Test 9: All-zero input with all-zero pattern - must match at every position
// =============================================================================

/// All-zero pattern in all-zero input must match at every position (overlapping).
#[test]
fn all_zero_pattern_all_zero_input() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x00\x00")
        .build()
        .unwrap();

    // 5 null bytes with 3-byte pattern = 3 overlapping matches
    let matches = ps.scan_overlapping(b"\x00\x00\x00\x00\x00").unwrap();

    assert_eq!(
        matches.len(),
        3,
        "CRITICAL: All-zero pattern should match at every position in all-zero input. \
        Expected 3 overlapping matches, got {}",
        matches.len()
    );

    for (i, m) in matches.iter().enumerate() {
        assert_eq!(
            m.start as usize, i,
            "CRITICAL: Match {} should start at position {}",
            i, i
        );
    }
}

/// Dense null byte matching.
#[test]
fn dense_null_byte_matching() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x00\x00\x00\x00")
        .build()
        .unwrap();

    // 10 null bytes with 5-byte pattern = 6 overlapping matches
    let input = vec![0u8; 10];
    let matches = ps.scan_overlapping(&input).unwrap();

    assert_eq!(
        matches.len(),
        6,
        "CRITICAL: Expected 6 overlapping null matches in 10-byte input, got {}",
        matches.len()
    );
}

/// Non-overlapping all-zero pattern matching.
#[test]
fn all_zero_non_overlapping() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x00")
        .build()
        .unwrap();

    // 6 null bytes with 2-byte pattern = 3 non-overlapping matches
    let matches = ps.scan(b"\x00\x00\x00\x00\x00\x00").unwrap();

    assert_eq!(
        matches.len(),
        3,
        "CRITICAL: Expected 3 non-overlapping matches, got {}",
        matches.len()
    );

    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 2);
    assert_eq!(matches[2].start, 4);
}

// =============================================================================
// Test 10: Binary input with null bytes - patterns must still be found
// =============================================================================

/// Pattern embedded in binary data with null bytes.
#[test]
fn pattern_in_binary_with_nulls() {
    let ps = PatternSet::builder()
        .literal_bytes(b"hello\x00world")
        .build()
        .unwrap();

    let data = b"prefix\x00\x00hello\x00world\x00suffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Pattern with embedded null should match in binary data"
    );
    assert_eq!(matches[0].start, 8);
    assert_eq!(matches[0].end, 19);
}

/// Multiple patterns in dense binary data.
#[test]
fn multiple_patterns_in_dense_binary() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x01\x02")
        .literal_bytes(b"\xFF\xFE\xFD")
        .literal_bytes(b"TARGET")
        .build()
        .unwrap();

    let data = b"\x00\x01\x02\x00\xFF\xFE\xFD\x00TARGET\x00";
    let matches = ps.scan(data).unwrap();

    assert_eq!(
        matches.len(),
        3,
        "CRITICAL: Expected 3 matches in binary data, found {}",
        matches.len()
    );

    // Verify each pattern was found
    let pattern_0 = matches.iter().any(|m| m.pattern_id == 0);
    let pattern_1 = matches.iter().any(|m| m.pattern_id == 1);
    let pattern_2 = matches.iter().any(|m| m.pattern_id == 2);

    assert!(pattern_0, "CRITICAL: Pattern 0 not found in binary data");
    assert!(pattern_1, "CRITICAL: Pattern 1 not found in binary data");
    assert!(pattern_2, "CRITICAL: Pattern 2 not found in binary data");
}

/// Pattern starting with null byte.
#[test]
fn pattern_starting_with_null() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00TARGET")
        .build()
        .unwrap();

    let data = b"prefix\x00TARGETsuffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Pattern starting with null should match"
    );
    assert_eq!(matches[0].start, 6);
}

/// Pattern ending with null byte.
#[test]
fn pattern_ending_with_null() {
    let ps = PatternSet::builder()
        .literal_bytes(b"TARGET\x00")
        .build()
        .unwrap();

    let data = b"prefixTARGET\x00suffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Pattern ending with null should match"
    );
}

// =============================================================================
// Test 11: Proptest - Zero false negatives
// For any pattern P and input containing P, scan MUST find P
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// ZERO FALSE NEGATIVES: If input contains pattern, scan MUST find it.
    /// This is the most critical property for security scanning.
    /// Fix: If this fails, malware will go undetected.
    #[test]
    fn prop_zero_false_negatives(
        prefix in vec(any::<u8>(), 0..100),
        pattern in vec(any::<u8>(), 1..64),
        suffix in vec(any::<u8>(), 0..100)
    ) {
        // Build input that definitely contains the pattern
        let mut input = prefix.clone();
        input.extend_from_slice(&pattern);
        input.extend_from_slice(&suffix);

        // Build pattern set
        let ps = match PatternSet::builder().literal_bytes(pattern.clone()).build() {
            Ok(p) => p,
            Err(_) => return Ok(()), // Skip invalid patterns
        };

        // Scan MUST find at least one match
        let matches = ps.scan(&input).unwrap();

        prop_assert!(
            !matches.is_empty(),
            "CRITICAL FALSE NEGATIVE: Pattern not found in input containing it! \
            Pattern length: {}, Input length: {}, Pattern bytes: {:?}",
            pattern.len(), input.len(), &pattern[..pattern.len().min(8)]
        );

        // Verify the match actually contains the pattern bytes
        if let Some(first_match) = matches.first() {
            let matched_bytes = &input[
                first_match.start as usize ..
                first_match.end as usize
            ];
            prop_assert_eq!(
                matched_bytes, pattern.as_slice(),
                "CRITICAL: Match bytes don't equal pattern bytes!"
            );
        }
    }

    /// ZERO FALSE NEGATIVES (overlapping): Same guarantee for overlapping mode.
    #[test]
    fn prop_zero_false_negatives_overlapping(
        prefix in vec(any::<u8>(), 0..50),
        pattern in vec(any::<u8>(), 1..32),
        suffix in vec(any::<u8>(), 0..50)
    ) {
        let mut input = prefix.clone();
        input.extend_from_slice(&pattern);
        input.extend_from_slice(&suffix);

        let ps = match PatternSet::builder().literal_bytes(pattern.clone()).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = ps.scan_overlapping(&input).unwrap();

        prop_assert!(
            !matches.is_empty(),
            "CRITICAL FALSE NEGATIVE (overlapping): Pattern not found!"
        );
    }

    /// Pattern must be found at exact position where it was inserted.
    #[test]
    fn prop_match_at_exact_position(
        prefix_len in 0usize..100,
        pattern in vec(any::<u8>(), 1..32)
    ) {
        let prefix: Vec<u8> = vec![b'A'; prefix_len];
        let suffix: Vec<u8> = vec![b'B'; 50];

        let mut input = prefix.clone();
        input.extend_from_slice(&pattern);
        input.extend_from_slice(&suffix);

        let ps = match PatternSet::builder().literal_bytes(pattern.clone()).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = ps.scan(&input).unwrap();

        // At least one match should start at the position where we inserted the pattern
        let expected_start = prefix_len as u32;
        let found_at_position = matches.iter().any(|m| m.start == expected_start);

        prop_assert!(
            found_at_position,
            "CRITICAL: Pattern not found at expected position {}! \
            Matches found at positions: {:?}",
            expected_start,
            matches.iter().map(|m| m.start).collect::<Vec<_>>()
        );
    }

    /// All-zero input with non-zero pattern should not produce false positives.
    #[test]
    fn prop_no_false_positives_all_zero_input(
        pattern in vec(any::<u8>(), 1..32)
    ) {
        // Skip if pattern is all zeros
        if pattern.iter().all(|&b| b == 0) {
            return Ok(());
        }

        let ps = match PatternSet::builder().literal_bytes(pattern).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        // All-zero input should not match non-zero pattern
        let input = vec![0u8; 1000];
        let matches = ps.scan(&input).unwrap();

        prop_assert!(
            matches.is_empty(),
            "CRITICAL FALSE POSITIVE: Non-zero pattern matched all-zero input!"
        );
    }

    /// Scanning twice produces identical results (idempotency).
    #[test]
    fn prop_scan_idempotent(
        input in vec(any::<u8>(), 0..1000),
        pattern in vec(any::<u8>(), 1..32)
    ) {
        let ps = match PatternSet::builder().literal_bytes(pattern).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches1 = ps.scan(&input).unwrap();
        let matches2 = ps.scan(&input).unwrap();

        prop_assert_eq!(
            matches1.len(), matches2.len(),
            "CRITICAL: Two scans produced different match counts!"
        );

        for (i, (m1, m2)) in matches1.iter().zip(matches2.iter()).enumerate() {
            prop_assert_eq!(
                m1.start, m2.start,
                "CRITICAL: Match {} start differs between scans", i
            );
            prop_assert_eq!(
                m1.end, m2.end,
                "CRITICAL: Match {} end differs between scans", i
            );
        }
    }
}

// =============================================================================
// Additional adversarial edge cases
// =============================================================================

/// Pattern with every possible byte value (0x00-0xFF).
#[test]
fn pattern_with_all_byte_values() {
    let mut builder = PatternSet::builder();

    // Add patterns for all 256 byte values
    for i in 0..=255u8 {
        builder = builder.literal_bytes(vec![i]);
    }

    let ps = builder.build().unwrap();

    // Input containing all byte values
    let data: Vec<u8> = (0..=255).collect();
    let mut out = vec![Match::from_parts(0, 0, 0); 512];
    let count = ps.scan_to_buffer(&data, &mut out).unwrap();

    assert_eq!(
        count, 256,
        "CRITICAL: Expected 256 matches for all byte values, got {}",
        count
    );
}

/// Very long pattern (10KB) must match correctly.
#[test]
fn very_long_pattern_ten_kb() {
    let pattern = "X".repeat(10_000);
    let ps = PatternSet::builder().literal(&pattern).build().unwrap();

    let mut data = String::new();
    data.push_str("prefix");
    data.push_str(&pattern);
    data.push_str("suffix");

    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: 10KB pattern should match exactly once"
    );
    assert_eq!(matches[0].start, 6);
    assert_eq!(matches[0].end, 10_006);
}

/// Pattern at every position in input - all must be found.
#[test]
fn pattern_at_every_position() {
    let ps = PatternSet::builder().literal("TARGET").build().unwrap();

    for pos in 0..=10 {
        let prefix = "A".repeat(pos);
        let input = format!("{}TARGET", prefix);
        let matches = ps.scan(input.as_bytes()).unwrap();

        assert_eq!(
            matches.len(),
            1,
            "CRITICAL: Pattern not found at position {}",
            pos
        );
        assert_eq!(
            matches[0].start as usize, pos,
            "CRITICAL: Pattern found at wrong position, expected {}",
            pos
        );
    }
}

/// Regex pattern in binary data with nulls.
#[test]
fn regex_in_binary_with_nulls() {
    let ps = PatternSet::builder()
        .regex(r"[a-z]+\x00+[a-z]+")
        .build()
        .unwrap();

    let data = b"prefixhello\x00\x00worldsuffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Regex should match pattern with null bytes"
    );
}

/// Case-insensitive matching must work with binary data.
#[test]
fn case_insensitive_binary_data() {
    let ps = PatternSet::builder()
        .literal("TeSt")
        .case_insensitive(true)
        .build()
        .unwrap();

    let matches = ps.scan(b"\x00test\x00TEST\x00").unwrap();

    assert_eq!(
        matches.len(),
        2,
        "CRITICAL: Case-insensitive scan should find both variations"
    );
}

/// Multiple adjacent matches with no gaps.
#[test]
fn adjacent_matches_no_gaps() {
    let ps = PatternSet::builder().literal("AB").build().unwrap();

    let matches = ps.scan(b"ABABABAB").unwrap();

    assert_eq!(
        matches.len(),
        4,
        "CRITICAL: Expected 4 adjacent matches, got {}",
        matches.len()
    );

    // Verify no gaps
    for i in 0..matches.len() - 1 {
        assert_eq!(
            matches[i].end,
            matches[i + 1].start,
            "CRITICAL: Gap between matches {} and {}: end={}, start={}",
            i,
            i + 1,
            matches[i].end,
            matches[i + 1].start
        );
    }
}

/// Boundary: Input exactly at 64KB threshold.
#[test]
fn input_at_64kb_boundary() {
    let ps = PatternSet::builder().literal("TARGET").build().unwrap();

    let mut data = vec![b'X'; 65_536];
    data[65_530..65_536].copy_from_slice(b"TARGET");

    let matches = ps.scan(&data).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "CRITICAL: Pattern not found at 64KB boundary"
    );
    assert_eq!(matches[0].start, 65_530);
}

/// Unicode pattern at byte boundaries.
#[test]
fn unicode_at_byte_boundaries() {
    // "日本語" is 9 bytes in UTF-8 (3 chars × 3 bytes each)
    let ps = PatternSet::builder().literal("日本語").build().unwrap();

    for prefix_len in 0..5 {
        let prefix = "A".repeat(prefix_len);
        let input = format!("{}日本語", prefix);
        let matches = ps.scan(input.as_bytes()).unwrap();

        assert_eq!(
            matches.len(),
            1,
            "CRITICAL: Unicode pattern not found at prefix_len={}",
            prefix_len
        );
        assert_eq!(
            matches[0].start as usize, prefix_len,
            "CRITICAL: Unicode match at wrong position"
        );
    }
}

/// Mixed literal and regex patterns - all must match.
#[test]
fn mixed_literal_and_regex_patterns() {
    let ps = PatternSet::builder()
        .literal("literal_target")
        .regex(r"regex[0-9]+target")
        .literal("another_literal")
        .regex(r"test[abc]pattern")
        .build()
        .unwrap();

    let data = b"literal_target here regex123target there another_literal and testbpattern";
    let matches = ps.scan(data).unwrap();

    assert_eq!(
        matches.len(),
        4,
        "CRITICAL: Expected 4 mixed matches, found {}",
        matches.len()
    );
}

/// scan_with visitor API must find same matches as scan().
#[test]
fn scan_with_finds_same_as_scan() {
    let ps = PatternSet::builder()
        .literal("target")
        .literal("other")
        .build()
        .unwrap();

    let data = b"target and other and target again";

    let scan_matches = ps.scan(data).unwrap();

    let mut with_matches = Vec::new();
    ps.scan_with(data, |m| {
        with_matches.push(m);
        true
    })
    .unwrap();

    assert_eq!(
        scan_matches.len(),
        with_matches.len(),
        "CRITICAL: scan_with found different number of matches than scan()"
    );
}

/// scan_count must equal scan().len().
#[test]
fn scan_count_equals_scan_len() {
    let ps = PatternSet::builder()
        .literal("ab")
        .literal("bc")
        .build()
        .unwrap();

    let data = b"abcabcabc";

    let matches = ps.scan(data).unwrap();
    let count = ps.scan_count(data).unwrap();

    assert_eq!(
        count,
        matches.len(),
        "CRITICAL: scan_count ({}) != scan().len() ({})",
        count,
        matches.len()
    );
}

/// HotSwapPatternSet must find matches after swap.
#[test]
fn hotswap_finds_matches_after_swap() {
    use warpstate::HotSwapPatternSet;

    let ps1 = PatternSet::builder()
        .literal("old_pattern")
        .build()
        .unwrap();
    let ps2 = PatternSet::builder()
        .literal("new_pattern")
        .build()
        .unwrap();

    let hotswap = HotSwapPatternSet::new(ps1);

    // Verify old pattern works
    let mut buf = [Match::from_parts(0, 0, 0); 10];
    let count1 = hotswap.scan(b"old_pattern", &mut buf).unwrap();
    assert_eq!(count1, 1, "CRITICAL: Initial pattern not found");

    // Swap and verify new pattern works
    let _old = hotswap.swap(ps2);
    let count2 = hotswap.scan(b"new_pattern", &mut buf).unwrap();
    assert_eq!(count2, 1, "CRITICAL: Pattern not found after swap");
}

/// CachedScanner must find same matches as PatternSet.scan().
#[test]
fn cached_scanner_parity() {
    use warpstate::CachedScanner;

    let ps = PatternSet::builder().literal("needle").build().unwrap();
    let scanner = CachedScanner::new(ps.ir()).unwrap();

    let data = b"needle in a needle stack";

    let ps_matches = ps.scan(data).unwrap();

    let mut cached_buf = [Match::from_parts(0, 0, 0); 10];
    let cached_count = scanner.scan(data, &mut cached_buf).unwrap();
    let cached_matches = &cached_buf[..cached_count];

    assert_eq!(
        ps_matches.len(),
        cached_matches.len(),
        "CRITICAL: CachedScanner found {} matches, PatternSet found {}",
        cached_matches.len(),
        ps_matches.len()
    );
}
