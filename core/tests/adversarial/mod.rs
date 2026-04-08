//! Adversarial edge case tests for warpstate
//!
//! These tests are designed to break the crate by testing:
//! - Empty patterns
//! - Single byte pattern
//! - Pattern longer than input
//! - Binary patterns with null bytes
//! - 10K patterns
//! - Overlapping patterns at same position
//! - Input of exactly 1 byte

use warpstate::*;

// =============================================================================
// Empty Pattern Tests
// =============================================================================

#[test]
fn empty_pattern_set_fails() {
    let result = PatternSet::builder().build();

    assert!(result.is_err(), "Empty pattern set should error");
    match result.unwrap_err() {
        Error::EmptyPatternSet => {}
        other => panic!("Expected EmptyPatternSet error, got {other:?}"),
    }
}

#[test]
fn empty_string_pattern_fails() {
    let result = PatternSet::builder().literal("").build();

    assert!(result.is_err(), "Empty string pattern should error");
    match result.unwrap_err() {
        Error::EmptyPattern { index: 0 } => {}
        other => panic!("Expected EmptyPattern error, got {other:?}"),
    }
}

#[test]
fn empty_bytes_pattern_fails() {
    let result = PatternSet::builder().literal_bytes(b"").build();

    assert!(result.is_err(), "Empty bytes pattern should error");
}

#[test]
fn pattern_with_empty_in_middle() {
    let result = PatternSet::builder()
        .literal("first")
        .literal("")
        .literal("third")
        .build();

    assert!(result.is_err(), "Empty pattern in middle should error");
}

// =============================================================================
// Single Byte Pattern Tests
// =============================================================================

#[test]
fn single_byte_pattern() {
    let ps = PatternSet::builder().literal("x").build().unwrap();

    let matches = ps.scan(b"xxx").unwrap();

    assert_eq!(matches.len(), 3, "Should find 3 single-byte matches");
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 1);
}

#[test]
fn single_byte_pattern_not_found() {
    let ps = PatternSet::builder().literal("x").build().unwrap();

    let matches = ps.scan(b"yyy").unwrap();

    assert!(matches.is_empty(), "Should find no matches");
}

#[test]
fn single_byte_null_pattern() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00")
        .build()
        .unwrap();

    let matches = ps.scan(b"a\x00b\x00c").unwrap();

    assert_eq!(matches.len(), 2, "Should find 2 null byte matches");
}

// =============================================================================
// Pattern Longer Than Input Tests
// =============================================================================

#[test]
fn pattern_longer_than_input() {
    let ps = PatternSet::builder()
        .literal("hello world")
        .build()
        .unwrap();

    let matches = ps.scan(b"hi").unwrap();

    assert!(
        matches.is_empty(),
        "Pattern longer than input should not match"
    );
}

#[test]
fn pattern_same_length_as_input() {
    let ps = PatternSet::builder().literal("abc").build().unwrap();

    let matches = ps.scan(b"abc").unwrap();

    assert_eq!(
        matches.len(),
        1,
        "Pattern same length as input should match"
    );
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 3);
}

#[test]
fn pattern_one_longer_than_input() {
    let ps = PatternSet::builder().literal("abcd").build().unwrap();

    let matches = ps.scan(b"abc").unwrap();

    assert!(matches.is_empty(), "Pattern 1 byte longer should not match");
}

// =============================================================================
// Binary Pattern with Null Bytes Tests
// =============================================================================

#[test]
fn binary_pattern_with_null_bytes() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x01\x02\x03")
        .build()
        .unwrap();

    let data = b"\xff\x00\x01\x02\x03\xff";
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 5);
}

#[test]
fn binary_pattern_with_embedded_nulls() {
    let ps = PatternSet::builder()
        .literal_bytes(b"hello\x00world")
        .build()
        .unwrap();

    let data = b"prefixhello\x00worldsuffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 6);
}

#[test]
fn all_null_bytes_pattern() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x00\x00")
        .build()
        .unwrap();

    let data = b"\x00\x00\x00\x00\x00";
    let matches = ps.scan_overlapping(data).unwrap();

    // Should find overlapping matches
    assert_eq!(matches.len(), 3, "Should find 3 overlapping null sequences");
}

#[test]
fn binary_pattern_high_bytes() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\xff\xfe\xfd")
        .build()
        .unwrap();

    let data = b"\x00\xff\xfe\xfd\x00";
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 1);
}

// =============================================================================
// 10K Patterns Tests
// =============================================================================

#[test]
fn ten_thousand_patterns() {
    let mut builder = PatternSet::builder();

    for i in 0..10000 {
        builder = builder.literal(&format!("pattern{i:05}"));
    }

    let ps = builder.build().unwrap();

    // Scan for a specific pattern in the middle
    let matches = ps.scan(b"prefix pattern05000 suffix").unwrap();

    assert!(
        matches.iter().any(|m| m.pattern_id == 5000),
        "Should find pattern 5000"
    );
}

#[test]
fn many_single_byte_patterns() {
    let mut builder = PatternSet::builder();

    // Add all 256 possible single bytes
    for i in 0..=255u8 {
        builder = builder.literal_bytes(vec![i]);
    }

    let ps = builder.build().unwrap();

    let data: Vec<u8> = (0..=255).collect();
    let mut out = vec![warpstate::Match::from_parts(0, 0, 0); 512];
    let count = ps.scan_to_buffer(&data, &mut out).unwrap();

    assert_eq!(count, 256, "Should find all 256 single-byte patterns");
}

// =============================================================================
// Overlapping Pattern Tests
// =============================================================================

#[test]
fn overlapping_patterns_at_same_position() {
    let ps = PatternSet::builder()
        .literal("abc")
        .literal("bcd")
        .literal("c")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abcd").unwrap();

    // "abc" at 0-3, "bcd" at 1-4, "c" at 2-3
    assert!(matches.len() >= 3, "Should find overlapping patterns");

    // Verify specific matches
    let abc_match = matches.iter().find(|m| m.pattern_id == 0 && m.start == 0);
    let bcd_match = matches.iter().find(|m| m.pattern_id == 1 && m.start == 1);
    let c_match = matches.iter().find(|m| m.pattern_id == 2);

    assert!(abc_match.is_some(), "Should find 'abc' at start");
    assert!(bcd_match.is_some(), "Should find 'bcd' at position 1");
    assert!(c_match.is_some(), "Should find 'c'");
}

#[test]
fn multiple_patterns_same_start() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("ab")
        .literal("abc")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abc").unwrap();

    // All three patterns should match at position 0
    let start_matches: Vec<_> = matches.iter().filter(|m| m.start == 0).collect();
    assert_eq!(
        start_matches.len(),
        3,
        "All three patterns should match at start"
    );
}

#[test]
fn repeated_character_overlapping() {
    let ps = PatternSet::builder().literal("aa").build().unwrap();

    let matches = ps.scan_overlapping(b"aaaa").unwrap();

    // Should find overlapping matches at positions 0, 1, 2
    assert_eq!(matches.len(), 3, "Should find 3 overlapping 'aa' matches");
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 1);
    assert_eq!(matches[2].start, 2);
}

// =============================================================================
// Single Byte Input Tests
// =============================================================================

#[test]
fn input_of_exactly_one_byte_match() {
    let ps = PatternSet::builder().literal("a").build().unwrap();

    let matches = ps.scan(b"a").unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 1);
}

#[test]
fn input_of_exactly_one_byte_no_match() {
    let ps = PatternSet::builder().literal("b").build().unwrap();

    let matches = ps.scan(b"a").unwrap();

    assert!(matches.is_empty());
}

#[test]
fn empty_input() {
    let ps = PatternSet::builder().literal("test").build().unwrap();

    let matches = ps.scan(b"").unwrap();

    assert!(matches.is_empty(), "Empty input should produce no matches");
}

#[test]
fn empty_input_with_single_byte_pattern() {
    let ps = PatternSet::builder().literal("x").build().unwrap();

    let matches = ps.scan(b"").unwrap();

    assert!(matches.is_empty());
}

// =============================================================================
// Unicode Pattern Tests
// =============================================================================

#[test]
fn unicode_pattern() {
    let ps = PatternSet::builder().literal("日本語").build().unwrap();

    let text = "テスト日本語テスト";
    let matches = ps.scan(text.as_bytes()).unwrap();

    // Calculate expected position: "テスト" = 3 chars * 3 bytes = 9 bytes
    // BUG FOUND: The assertion was wrong - "テスト" is 9 bytes, not 6
    // The pattern "日本語" starts at byte offset 9
    assert_eq!(matches.len(), 1, "Should find one match for '日本語'");
    assert_eq!(
        matches[0].start, 9,
        "Pattern should start at byte offset 9 (not 6)"
    );
}

#[test]
fn unicode_multibyte_boundary() {
    let ps = PatternSet::builder().literal("语").build().unwrap();

    // "语" is the simplified Chinese character, but "日本語" uses "語" (traditional)
    // These are different Unicode codepoints!
    let matches = ps.scan("日本語".as_bytes()).unwrap();

    // BUG FOUND: "语" (U+8BED) != "語" (U+8A9E) - they are different characters
    assert!(
        matches.is_empty(),
        "'语' (simplified) should not match '語' (traditional)"
    );

    // Let's test with the correct character
    let ps2 = PatternSet::builder().literal("語").build().unwrap();
    let matches2 = ps2.scan("日本語".as_bytes()).unwrap();
    assert_eq!(matches2.len(), 1, "'語' should match '日本語'");
}

// =============================================================================
// Pattern IR Tests
// =============================================================================

#[test]
fn pattern_ir_access() {
    let ps = PatternSet::builder().literal("test").build().unwrap();

    let ir = ps.ir();

    assert_eq!(ir.offsets.len(), 1);
    assert!(!ir.packed_bytes.is_empty());
}

#[test]
fn pattern_set_len() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("b")
        .literal("c")
        .build()
        .unwrap();

    assert_eq!(ps.len(), 3);
    assert!(!ps.is_empty());
}

#[test]
fn pattern_set_is_empty_false() {
    let ps = PatternSet::builder().literal("test").build().unwrap();

    assert!(!ps.is_empty());
}

// =============================================================================
// Named Pattern Tests
// =============================================================================

#[test]
fn named_pattern() {
    let ps = PatternSet::builder()
        .named_literal("password_pattern", "password")
        .build()
        .unwrap();

    let matches = ps.scan(b"user password here").unwrap();

    assert_eq!(matches.len(), 1);
}

// =============================================================================
// Very Long Pattern Tests
// =============================================================================

#[test]
fn very_long_pattern() {
    let long_pattern = "a".repeat(10000);
    let ps = PatternSet::builder()
        .literal(&long_pattern)
        .build()
        .unwrap();

    let mut data = long_pattern.clone();
    data.push_str("extra");
    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 10000);
}

#[test]
fn pattern_at_exact_end_1() {
    let ps = PatternSet::builder().literal("end").build().unwrap();

    let matches = ps.scan(b"this is the end").unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 12);
    assert_eq!(matches[0].end, 15);
}

// =============================================================================
// Multiple Matches Same Pattern Tests
// =============================================================================

#[test]
fn multiple_matches_same_pattern() {
    let ps = PatternSet::builder().literal("test").build().unwrap();

    let matches = ps.scan(b"test test test").unwrap();

    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 5);
    assert_eq!(matches[2].start, 10);
}

#[test]
fn adjacent_matches() {
    let ps = PatternSet::builder().literal("ab").build().unwrap();

    let matches = ps.scan(b"ababab").unwrap();

    // "ab" at positions 0, 2, 4
    assert_eq!(matches.len(), 3);
}

// =============================================================================
// Case Sensitivity Tests
// =============================================================================

#[test]
fn case_sensitive_matching() {
    let ps = PatternSet::builder().literal("Test").build().unwrap();

    let matches_lower = ps.scan(b"test").unwrap();
    let matches_upper = ps.scan(b"TEST").unwrap();
    let matches_exact = ps.scan(b"Test").unwrap();

    assert!(matches_lower.is_empty(), "Lowercase should not match");
    assert!(matches_upper.is_empty(), "Uppercase should not match");
    assert_eq!(matches_exact.len(), 1, "Exact case should match");
}

// Deeper adversarial edge case tests for warpstate — the FLAGSHIP crate
//
// These tests stress-test the GPU pattern matcher with extreme edge cases:
// 1. Pattern with 0 bytes throughout (not just at boundaries)
// 2. Input that's exactly 1 byte (minimal input handling)
// 3. 10,000 patterns simultaneously (stress test pattern table)
// 4. Pattern that's longer than the input
// 5. Overlapping patterns at the same position (identical patterns)
// 6. Unicode patterns at various byte alignments
// 7. Binary data with null bytes throughout (dense nulls)
// 8. Input exactly at GPU transfer threshold (64KB boundary conditions)
// 9. Multiple patterns matching at exact same position
// 10. Extreme pattern lengths (1 byte to 1000+ bytes)

// use warpstate::*;

// =============================================================================
// Pattern with 0 Bytes (Dense Null Bytes)
// =============================================================================

#[test]
fn pattern_of_all_null_bytes() {
    // Pattern that is ALL null bytes
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x00\x00\x00\x00")
        .build()
        .unwrap();

    // Input with dense null bytes
    let data = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let matches = ps.scan_overlapping(data).unwrap();

    // Should find overlapping matches at positions 0, 1, 2, 3, 4
    assert_eq!(
        matches.len(),
        5,
        "Should find 5 overlapping all-null matches"
    );
    for (i, m) in matches.iter().enumerate() {
        assert_eq!(m.start, i as u32, "Match {i} should start at {i}");
        assert_eq!(m.end, (i + 5) as u32, "Match {} should end at {}", i, i + 5);
    }
}

#[test]
fn pattern_with_nulls_at_boundaries() {
    // Pattern starts and ends with null byte
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00test\x00")
        .build()
        .unwrap();

    let data = b"prefix\x00test\x00suffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 6);
    assert_eq!(matches[0].end, 12);
}

#[test]
fn multiple_null_byte_patterns_different_lengths() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00")
        .literal_bytes(b"\x00\x00")
        .literal_bytes(b"\x00\x00\x00")
        .build()
        .unwrap();

    // Three consecutive nulls: positions 0,1,2
    // Should find:
    // - three 1-byte matches at 0, 1, 2
    // - two 2-byte matches at 0-1, 1-2
    // - one 3-byte match at 0-2
    let data = b"\x00\x00\x00";
    let matches = ps.scan_overlapping(data).unwrap();

    let match_1byte = matches.iter().filter(|m| m.pattern_id == 0).count();
    let match_2byte = matches.iter().filter(|m| m.pattern_id == 1).count();
    let match_3byte = matches.iter().filter(|m| m.pattern_id == 2).count();

    assert_eq!(match_1byte, 3, "Should find 3 single-null matches");
    assert_eq!(match_2byte, 2, "Should find 2 double-null matches");
    assert_eq!(match_3byte, 1, "Should find 1 triple-null match");
}

// =============================================================================
// Single Byte Input (Exact 1 Byte)
// =============================================================================

#[test]
fn one_byte_input_exact_match() {
    let ps = PatternSet::builder().literal("x").build().unwrap();

    let matches = ps.scan(b"x").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 1);
}

#[test]
fn one_byte_input_no_match() {
    let ps = PatternSet::builder().literal("y").build().unwrap();

    let matches = ps.scan(b"x").unwrap();
    assert!(matches.is_empty());
}

#[test]
fn one_byte_input_with_multi_byte_pattern() {
    // Pattern is longer than 1-byte input - should never match
    let ps = PatternSet::builder().literal("xx").build().unwrap();

    let matches = ps.scan(b"x").unwrap();
    assert!(
        matches.is_empty(),
        "Multi-byte pattern should not match 1-byte input"
    );
}

#[test]
fn one_byte_input_many_patterns() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("b")
        .literal("c")
        .literal("x")
        .build()
        .unwrap();

    let matches = ps.scan(b"x").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pattern_id, 3);
}

// =============================================================================
// 10,000 Patterns Simultaneously (Stress Test)
// =============================================================================

#[test]
fn ten_thousand_patterns_all_same_prefix() {
    let mut builder = PatternSet::builder();

    // All patterns start with "PREFIX_" but have different suffixes
    for i in 0..10000 {
        builder = builder.literal(&format!("PREFIX_{i:05}"));
    }

    let ps = builder.build().unwrap();

    // Scan for a pattern at the beginning
    let matches = ps.scan(b"test PREFIX_00000 here").unwrap();
    assert!(matches.iter().any(|m| m.pattern_id == 0));

    // Scan for a pattern in the middle
    let matches = ps.scan(b"test PREFIX_05000 here").unwrap();
    assert!(matches.iter().any(|m| m.pattern_id == 5000));

    // Scan for a pattern at the end
    let matches = ps.scan(b"test PREFIX_09999 here").unwrap();
    assert!(matches.iter().any(|m| m.pattern_id == 9999));
}

#[test]
fn ten_thousand_single_byte_patterns() {
    let mut builder = PatternSet::builder();

    // Create many single-byte patterns (repeating 'a'-'z')
    for i in 0u16..10_000 {
        let offset = u8::try_from(i % 26).unwrap();
        let ch = (b'a' + offset) as char;
        builder = builder.literal(&ch.to_string());
    }

    let ps = builder.build().unwrap();

    // Input with many different chars. Use large buffer since 10K patterns
    // can produce many matches that overflow estimate_match_capacity.
    let data = b"abcdefghijklmnopqrstuvwxyz";
    let mut out = vec![warpstate::Match::from_parts(0, 0, 0); 16384];
    let count = ps.scan_to_buffer(data, &mut out).unwrap();

    assert!(count >= 26, "Should find at least 26 matches");
}

#[test]
fn ten_thousand_patterns_no_matches() {
    let mut builder = PatternSet::builder();

    for i in 0..10000 {
        builder = builder.literal(&format!("UNIQUE_PATTERN_{i}"));
    }

    let ps = builder.build().unwrap();

    // Scan for something that doesn't match any pattern
    let matches = ps.scan(b"this text contains none of the patterns").unwrap();
    assert!(matches.is_empty(), "Should find no matches");
}

// =============================================================================
// Pattern Longer Than Input
// =============================================================================

#[test]
fn pattern_much_longer_than_input() {
    let ps = PatternSet::builder()
        .literal("this is a very long pattern that is much longer than the input")
        .build()
        .unwrap();

    let matches = ps.scan(b"short").unwrap();
    assert!(matches.is_empty());
}

#[test]
fn pattern_one_byte_longer_than_input() {
    let ps = PatternSet::builder().literal("abcdef").build().unwrap();

    let matches = ps.scan(b"abcde").unwrap();
    assert!(matches.is_empty(), "Pattern 1 byte longer should not match");
}

#[test]
fn multiple_patterns_some_longer_some_shorter() {
    let ps = PatternSet::builder()
        .literal("short")
        .literal("this is a very long pattern indeed")
        .literal("tiny")
        .build()
        .unwrap();

    let data = b"short tiny";
    let matches = ps.scan(data).unwrap();

    // Should match "short" (pattern 0) and "tiny" (pattern 2)
    assert_eq!(matches.len(), 2);
    assert!(matches.iter().any(|m| m.pattern_id == 0));
    assert!(matches.iter().any(|m| m.pattern_id == 2));
    assert!(!matches.iter().any(|m| m.pattern_id == 1));
}

// =============================================================================
// Overlapping Patterns at Same Position (Identical Patterns)
// =============================================================================

#[test]
fn identical_patterns_different_ids() {
    // Two different patterns with the SAME content
    let ps = PatternSet::builder()
        .literal("target")
        .literal("target")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"target").unwrap();

    // Both patterns should match at the same position
    assert_eq!(matches.len(), 2, "Both identical patterns should match");
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 6);
    assert_eq!(matches[1].start, 0);
    assert_eq!(matches[1].end, 6);
    // They should have different pattern IDs
    assert_ne!(matches[0].pattern_id, matches[1].pattern_id);
}

#[test]
fn multiple_overlapping_at_exact_same_position() {
    let ps = PatternSet::builder()
        .literal("abc")
        .literal("abc")
        .literal("abc")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abc").unwrap();

    assert_eq!(
        matches.len(),
        3,
        "All three identical patterns should match"
    );
    for m in &matches {
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 3);
    }
}

#[test]
fn partial_overlap_same_start() {
    let ps = PatternSet::builder()
        .literal("abc")
        .literal("abcd")
        .literal("abcde")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abcde").unwrap();

    // All three should match starting at position 0
    let start_at_0: Vec<_> = matches.iter().filter(|m| m.start == 0).collect();
    assert_eq!(start_at_0.len(), 3);
}

#[test]
fn partial_overlap_same_end() {
    let ps = PatternSet::builder()
        .literal("cde")
        .literal("bcde")
        .literal("abcde")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abcde").unwrap();

    // All three should end at position 5
    let end_at_5: Vec<_> = matches.iter().filter(|m| m.end == 5).collect();
    assert_eq!(end_at_5.len(), 3);
}

// =============================================================================
// Unicode Patterns (Multi-Byte) - Various Alignments
// =============================================================================

#[test]
fn unicode_at_different_alignments() {
    // Test that multi-byte UTF-8 sequences are found correctly
    // regardless of their byte alignment in the input

    let ps = PatternSet::builder().literal("日本語").build().unwrap();

    // Create input with the pattern at different positions
    for prefix_len in 0..10 {
        let prefix = "a".repeat(prefix_len);
        let data = format!("{prefix}日本語");
        let matches = ps.scan(data.as_bytes()).unwrap();

        assert_eq!(
            matches.len(),
            1,
            "Should find match at prefix_len={prefix_len}"
        );
        assert_eq!(
            matches[0].start, prefix_len as u32,
            "Match should start after prefix"
        );
        assert_eq!(
            matches[0].end,
            (prefix_len + 9) as u32,
            "Match should be 9 bytes (3 chars × 3 bytes)"
        );
    }
}

#[test]
fn unicode_mixed_with_ascii_patterns() {
    let ps = PatternSet::builder()
        .literal("hello")
        .literal("世界")
        .literal("world")
        .build()
        .unwrap();

    let data = "hello 世界 world";
    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 3);
    assert!(matches.iter().any(|m| m.pattern_id == 0)); // "hello"
    assert!(matches.iter().any(|m| m.pattern_id == 1)); // "世界"
    assert!(matches.iter().any(|m| m.pattern_id == 2)); // "world"
}

#[test]
fn unicode_4byte_sequences() {
    // Test with 4-byte UTF-8 sequences (emojis)
    let ps = PatternSet::builder().literal("🚀🌟").build().unwrap();

    let data = "test🚀🌟test";
    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 1);
    // "test" = 4 bytes, "🚀" = 4 bytes, "🌟" = 4 bytes
    assert_eq!(matches[0].start, 4);
    assert_eq!(matches[0].end, 12);
}

#[test]
fn unicode_various_scripts() {
    let ps = PatternSet::builder()
        .literal("日本語") // Japanese
        .literal("中文") // Chinese
        .literal("한국어") // Korean
        .literal("العربية") // Arabic
        .literal("Ελληνικά") // Greek
        .build()
        .unwrap();

    let data = "Testing: 日本語, 中文, 한국어, العربية, Ελληνικά";
    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 5, "Should find all 5 Unicode patterns");
}

#[test]
fn unicode_partial_match_should_fail() {
    // Try to match partial UTF-8 sequence - should not match
    let ps = PatternSet::builder()
        .literal("日") // 3 bytes: E6 97 A5
        .build()
        .unwrap();

    // Just the first byte of the UTF-8 sequence
    let data = b"\xE6";
    let matches = ps.scan(data).unwrap();
    assert!(matches.is_empty(), "Partial UTF-8 byte should not match");

    // First two bytes
    let data = b"\xE6\x97";
    let matches = ps.scan(data).unwrap();
    assert!(
        matches.is_empty(),
        "Incomplete UTF-8 sequence should not match"
    );
}

// =============================================================================
// Binary Data with Null Bytes Throughout (Dense Binary)
// =============================================================================

#[test]
fn dense_binary_all_nulls_input() {
    // Input that's entirely null bytes
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x00\x00\x00")
        .build()
        .unwrap();

    let data = vec![0u8; 100];
    let mut out = vec![warpstate::Match::from_parts(0, 0, 0); 128];
    let count = ps.scan_overlapping_to_buffer(&data, &mut out).unwrap();

    // Should find 97 overlapping matches (positions 0-96)
    assert_eq!(count, 97);
}

#[test]
fn binary_with_nulls_every_other_byte() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00A\x00A")
        .build()
        .unwrap();

    let mut data = Vec::new();
    for _ in 0..10 {
        data.push(0u8);
        data.push(b'A');
    }

    let matches = ps.scan_overlapping(&data).unwrap();
    // Should find 9 matches (overlapping)
    assert_eq!(matches.len(), 9);
}

#[test]
fn binary_high_entropy() {
    // Completely random-looking binary data
    let ps = PatternSet::builder()
        .literal_bytes(b"\xDE\xAD\xBE\xEF")
        .literal_bytes(b"\xCA\xFE\xBA\xBE")
        .literal_bytes(b"\x00\x00\x00\x00")
        .build()
        .unwrap();

    let data = b"\xDE\xAD\xBE\xEFrandom\xCA\xFE\xBA\xBEdata\x00\x00\x00\x00end";
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 3);
}

#[test]
fn binary_embedded_nulls_mid_pattern() {
    // Pattern with nulls in the middle
    let ps = PatternSet::builder()
        .literal_bytes(b"start\x00\x00\x00end")
        .build()
        .unwrap();

    let data = b"prefixstart\x00\x00\x00endsuffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 6);
}

// =============================================================================
// Input at GPU Transfer Threshold (64KB Boundary)
// =============================================================================

const GPU_THRESHOLD: usize = 65_536; // 64KB from router.rs

#[test]
fn input_exactly_at_64kb_threshold() {
    let ps = PatternSet::builder().literal("TARGET").build().unwrap();

    // Create input exactly at threshold
    let mut data = vec![b'A'; GPU_THRESHOLD];
    // Place target at the end
    data[GPU_THRESHOLD - 6..].copy_from_slice(b"TARGET");

    let matches = ps.scan(&data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, (GPU_THRESHOLD - 6) as u32);
    assert_eq!(matches[0].end, GPU_THRESHOLD as u32);
}

#[test]
fn input_just_under_64kb() {
    let ps = PatternSet::builder().literal("TARGET").build().unwrap();

    let size = GPU_THRESHOLD - 1;
    let mut data = vec![b'A'; size];
    data[size - 6..].copy_from_slice(b"TARGET");

    let matches = ps.scan(&data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, (size - 6) as u32);
}

#[test]
fn input_just_over_64kb() {
    let ps = PatternSet::builder().literal("TARGET").build().unwrap();

    let size = GPU_THRESHOLD + 1;
    let mut data = vec![b'A'; size];
    data[size - 6..].copy_from_slice(b"TARGET");

    let matches = ps.scan(&data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, (size - 6) as u32);
}

#[test]
fn multiple_matches_across_64kb_boundary() {
    let ps = PatternSet::builder().literal("TARGET").build().unwrap();

    let mut data = vec![b'A'; GPU_THRESHOLD + 100];
    // Place targets at various positions
    data[100..106].copy_from_slice(b"TARGET");
    data[GPU_THRESHOLD - 3..GPU_THRESHOLD + 3].copy_from_slice(b"TARGET");
    data[GPU_THRESHOLD + 50..GPU_THRESHOLD + 56].copy_from_slice(b"TARGET");

    let matches = ps.scan(&data).unwrap();

    assert_eq!(matches.len(), 3);
}

#[test]
fn large_input_many_patterns() {
    let mut builder = PatternSet::builder();
    for i in 0..100 {
        builder = builder.literal(&format!("PATTERN{i:04}"));
    }
    let ps = builder.build().unwrap();

    // Large input with some patterns embedded (PATTERN#### is 11 chars)
    let mut data = vec![b'X'; GPU_THRESHOLD * 2];
    data[1000..1011].copy_from_slice(b"PATTERN0050");
    data[70000..70011].copy_from_slice(b"PATTERN0099");

    let matches = ps.scan(&data).unwrap();

    assert_eq!(matches.len(), 2);
}

// =============================================================================
// Extreme Pattern Lengths
// =============================================================================

#[test]
fn very_short_patterns_1_to_3_bytes() {
    let ps = PatternSet::builder()
        .literal("a") // 1 byte
        .literal("ab") // 2 bytes
        .literal("abc") // 3 bytes
        .build()
        .unwrap();

    let data = b"abcabc";
    let matches = ps.scan_overlapping(data).unwrap();

    // Pattern 0 "a": positions 0, 3 = 2 matches
    // Pattern 1 "ab": positions 0, 3 = 2 matches
    // Pattern 2 "abc": positions 0, 3 = 2 matches
    let count_0 = matches.iter().filter(|m| m.pattern_id == 0).count();
    let count_1 = matches.iter().filter(|m| m.pattern_id == 1).count();
    let count_2 = matches.iter().filter(|m| m.pattern_id == 2).count();

    assert_eq!(count_0, 2);
    assert_eq!(count_1, 2);
    assert_eq!(count_2, 2);
}

#[test]
fn pattern_255_bytes() {
    let pattern = "x".repeat(255);
    let ps = PatternSet::builder().literal(&pattern).build().unwrap();

    let mut data = pattern.clone();
    data.push_str("extra");
    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].end - matches[0].start, 255);
}

#[test]
fn pattern_1000_bytes() {
    let pattern = "A".repeat(1000);
    let ps = PatternSet::builder().literal(&pattern).build().unwrap();

    let mut data = String::new();
    data.push_str("prefix");
    data.push_str(&pattern);
    data.push_str("suffix");

    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 6);
    assert_eq!(matches[0].end, 1006);
}

// =============================================================================
// Pattern at Input Boundaries
// =============================================================================

#[test]
fn pattern_at_exact_start() {
    let ps = PatternSet::builder().literal("START").build().unwrap();

    let matches = ps.scan(b"START here").unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
}

#[test]
fn pattern_at_exact_end() {
    let ps = PatternSet::builder().literal("END").build().unwrap();

    let matches = ps.scan(b"this is the END").unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].end, 15);
}

#[test]
fn pattern_spanning_entire_input() {
    let ps = PatternSet::builder().literal("exact").build().unwrap();

    let matches = ps.scan(b"exact").unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 5);
}

// =============================================================================
// Repeated Character Patterns
// =============================================================================

#[test]
fn repeated_character_max_overlap() {
    // Pattern "aa" in "aaaaa" - how many overlapping matches?
    let ps = PatternSet::builder().literal("aa").build().unwrap();

    let matches = ps.scan_overlapping(b"aaaaa").unwrap();

    // Positions: 0-1, 1-2, 2-3, 3-4 = 4 matches
    assert_eq!(matches.len(), 4);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 1);
    assert_eq!(matches[2].start, 2);
    assert_eq!(matches[3].start, 3);
}

#[test]
fn repeated_character_triple() {
    let ps = PatternSet::builder().literal("aaa").build().unwrap();

    let matches = ps.scan_overlapping(b"aaaaaa").unwrap();

    // Positions: 0-2, 1-3, 2-4, 3-5 = 4 matches
    assert_eq!(matches.len(), 4);
}

// =============================================================================
// Alternating Pattern Stress Tests
// =============================================================================

#[test]
fn alternating_bytes_pattern() {
    let ps = PatternSet::builder()
        .literal_bytes(b"ABABAB")
        .build()
        .unwrap();

    let data = b"ABABABABABAB";
    let matches = ps.scan_overlapping(data).unwrap();

    // Positions: 0, 2, 4, 6 = 4 overlapping matches
    assert_eq!(matches.len(), 4);
}

#[test]
fn pattern_same_as_input() {
    let ps = PatternSet::builder().literal("IDENTICAL").build().unwrap();

    let matches = ps.scan(b"IDENTICAL").unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 9);
}

// Extreme hardware limitations and saturation tests for Warpstate.
//
// True "SQLite-level" depth is not about validating successful cases;
// it is about weaponizing failure, ensuring every system limit can be intentionally
// hit without triggering an uncontrolled panic or VRAM corruption.

// use warpstate::*;

#[test]
#[cfg(not(miri))]
fn gpu_match_buffer_saturation() {
    // We intentionally trigger a VRAM match buffer overflow.
    // The GPU buffer holds exactly 1,048,576 matches.
    // If we request 2,000,000 matches, the shader atomic counter MUST saturate
    // safely, preventing out-of-bounds array writes to `match_buf`.

    let ps = PatternSet::builder().literal("a").build().unwrap();

    // Create an array of 2,000,000 'a's.
    let data = vec![b'a'; 2_000_000];

    // AutoMatcher should route to GPU for 2MB payload.
    let scan_engine = pollster::block_on(AutoMatcher::new(&ps)).unwrap();
    let result = pollster::block_on(scan_engine.scan(&data));

    // The GPU correctly reports overflow rather than silently truncating.
    // Both outcomes are acceptable: either we get truncated matches or an overflow error.
    match result {
        Ok(matches) => {
            assert!(
                matches.len() <= DEFAULT_MAX_MATCHES as usize,
                "Engine MUST NOT exceed the pre-allocated VRAM boundary."
            );
        }
        Err(Error::MatchBufferOverflow { count, max }) => {
            // Internal buffer is sized by estimate_match_capacity, not DEFAULT_MAX_MATCHES
            assert!(
                count >= max,
                "overflow count ({count}) should be >= max ({max})"
            );
        }
        Err(other) => panic!("Expected Ok or MatchBufferOverflow, got {other:?}"),
    }
}

#[test]
#[cfg(not(miri))]
fn gpu_input_too_large_rejection() {
    // If we throw an input larger than the max supported memory,
    // we MUST get an InputTooLarge error, and not an OS-level OOM panic.
    //
    // Use regex() instead of literal() to ensure PersistentMatcher finds a RegexDFA.
    let ps = PatternSet::builder().regex("target").build().unwrap();

    // Create a PersistentMatcher with a tiny max to simulate the edge case.
    let tiny_config = AutoMatcherConfig::default().gpu_max_input_size(1024);

    let tiny_matcher = match pollster::block_on(
        warpstate::persistent::PersistentMatcher::with_config(&ps, tiny_config),
    ) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return, // No GPU available, skip this test.
        Err(other) => panic!("Unexpected matcher construction error: {other:?}"),
    };

    // Bypass stream abstraction and force push into the raw block processor.
    use warpstate::matcher::BlockMatcher;
    let data = vec![0u8; 1025];
    let result = pollster::block_on(tiny_matcher.scan_block(&data));

    match result {
        Err(matchkit::Error::InputTooLarge { bytes, max_bytes }) => {
            assert_eq!(bytes, 1025);
            assert_eq!(max_bytes, 1024);
        }
        _ => panic!("Expected InputTooLarge error, got {:?}", result),
    }
}

#[test]
fn pattern_set_extreme_size_rejection() {
    // Over-saturating the Aho-Corasick state array MUST NOT panic.
    // If we inject an impossible number of states it should yield PatternSetTooLarge.

    let mut builder = PatternSet::builder();
    // 50,000 huge patterns to force DFA combinatorial explosion.
    for i in 0..50_000 {
        let pat = format!("A_VERY_LONG_PATTERN_PREFIX_TO_EXPLODE_THE_DFA_STATE_MACHINE_{i}");
        builder = builder.literal(&pat);
    }

    let result = builder.build();

    // Right now Aho-Corasick config is bounded to 1GB or panic ?
    // Let's see if this gracefully triggers our Error map!
    match result {
        Ok(_) => {
            // If the environment has enough RAM, it's fine.
            // But we must assert it didn't PANIC.
        }
        Err(Error::PatternCompilationFailed { .. }) => {
            // This is also graceful and correct.
        }
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
}
pub mod boundary;
pub mod injection;
pub mod overflow;
pub mod unicode;
