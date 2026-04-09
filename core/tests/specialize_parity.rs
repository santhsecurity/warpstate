//! Parity tests for pattern-set specialization strategies.
//!
//! Each test verifies that the specialized fast path produces identical results
//! to the standard Aho-Corasick/DFA path.

use warpstate::{PatternSet, ScanStrategy};

/// Reference scan using the default (non-specialized) path.
///
/// Forces Aho-Corasick/DFA by bypassing the specialization selection.
fn reference_scan(patterns: &PatternSet, data: &[u8]) -> Vec<warpstate::Match> {
    // Always use the Aho-Corasick path for comparison.
    // Buffer must be larger than the maximum possible matches to avoid overflow.
    let capacity = data.len().max(64) + 1;
    let mut out_matches = vec![warpstate::Match::from_parts(0, 0, 0); capacity];
    let count = warpstate::cpu::scan(patterns.ir(), data, &mut out_matches).unwrap();
    out_matches[..count].to_vec()
}

/// Verify that SingleMemchr strategy produces identical results to the default path.
#[test]
fn single_memchr_parity_with_default() {
    let patterns = PatternSet::builder().literal("needle").build().unwrap();

    // Verify strategy selection
    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::SingleMemchr { .. }),
        "Expected SingleMemchr strategy"
    );

    // Test various inputs
    let test_cases = [
        b"".as_slice(),
        b"no match here",
        b"needle",
        b"needle at the start",
        b"ends with needle",
        b"needle in the middle needle",
        b"needle needle needle",
        b"NEEDLE (case sensitive)",
        b"needl (partial)",
        b"xxneedlexxneedlexx",
    ];

    for data in &test_cases {
        let expected = reference_scan(&patterns, data);
        let actual = patterns.scan(data).unwrap();
        assert_eq!(
            expected,
            actual,
            "Mismatch for input: {:?}",
            String::from_utf8_lossy(data)
        );
    }
}

/// Verify SingleMemchr with longer literal (up to 32 bytes).
#[test]
fn single_memchr_32byte_literal_parity() {
    let needle = "a".repeat(32);
    let patterns = PatternSet::builder().literal(&needle).build().unwrap();

    // Verify strategy selection (should still be SingleMemchr)
    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::SingleMemchr { .. }),
        "Expected SingleMemchr for 32-byte literal"
    );

    let data = format!("{0} middle {0}", needle);
    let expected = reference_scan(&patterns, data.as_bytes());
    let actual = patterns.scan(data.as_bytes()).unwrap();
    assert_eq!(expected, actual);
}

/// Verify SingleMemchr falls back to AhoCorasick for literals > 32 bytes.
#[test]
fn long_literal_uses_aho_corasick() {
    let needle = "a".repeat(33);
    let patterns = PatternSet::builder().literal(&needle).build().unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::AhoCorasick),
        "Expected AhoCorasick for >32-byte literal, got {:?}",
        strategy
    );

    let data = format!("{0} middle {0}", needle);
    let expected = reference_scan(&patterns, data.as_bytes());
    let actual = patterns.scan(data.as_bytes()).unwrap();
    assert_eq!(expected, actual);
}

/// Verify MultiMemchr strategy produces identical results to the default path.
#[test]
fn multi_memchr_parity_with_default() {
    let patterns = PatternSet::builder()
        .literal("foo")
        .literal("bar")
        .literal("baz")
        .build()
        .unwrap();

    // Verify strategy selection
    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::MultiMemchr { .. }),
        "Expected MultiMemchr strategy"
    );

    let test_cases = [
        b"".as_slice(),
        b"no matches",
        b"foo",
        b"bar",
        b"baz",
        b"foo bar baz",
        b"foofoofoo",
        b"foobarbaz",
        b"xxfooyybazz",
        b"foo bar baz foo bar baz",
        b"FOO (case sensitive)",
        b"fo (partial match)",
    ];

    for data in &test_cases {
        let expected = reference_scan(&patterns, data);
        let actual = patterns.scan(data).unwrap();
        assert_eq!(
            expected,
            actual,
            "Mismatch for input: {:?}",
            String::from_utf8_lossy(data)
        );
    }
}

/// Verify MultiMemchr with boundary conditions (exactly 8 patterns).
#[test]
fn multi_memchr_max_patterns_parity() {
    let patterns = PatternSet::builder()
        .literal("p1")
        .literal("p2")
        .literal("p3")
        .literal("p4")
        .literal("p5")
        .literal("p6")
        .literal("p7")
        .literal("p8")
        .build()
        .unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::MultiMemchr { .. }),
        "Expected MultiMemchr for exactly 8 patterns"
    );

    let data = b"p1 p2 p3 p4 p5 p6 p7 p8 p1";
    let expected = reference_scan(&patterns, data);
    let actual = patterns.scan(data).unwrap();
    assert_eq!(expected, actual);
}

/// Verify 9 patterns uses AhoCorasick instead of MultiMemchr.
#[test]
fn nine_literals_uses_aho_corasick() {
    let mut builder = PatternSet::builder();
    for i in 1..=9 {
        builder = builder.literal(format!("p{}", i));
    }
    let patterns = builder.build().unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::AhoCorasick),
        "Expected AhoCorasick for 9 literals"
    );

    let data = b"p1 p2 p3 p4 p5 p6 p7 p8 p9";
    let expected = reference_scan(&patterns, data);
    let actual = patterns.scan(data).unwrap();
    assert_eq!(expected, actual);
}

/// Verify MultiMemchr falls back to AhoCorasick if any literal > 16 bytes.
#[test]
fn multi_memchr_long_literal_fallback() {
    let patterns = PatternSet::builder()
        .literal("short")
        .literal("a".repeat(17)) // Just over the 16-byte limit
        .build()
        .unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::AhoCorasick),
        "Expected AhoCorasick when one literal > 16 bytes"
    );

    let data = b"short aaaaaaaaaaaaaaaaa";
    let expected = reference_scan(&patterns, data);
    let actual = patterns.scan(data).unwrap();
    assert_eq!(expected, actual);
}

/// Verify MultiMemchr with exactly 16-byte literals.
#[test]
fn multi_memchr_16byte_literals_parity() {
    let patterns = PatternSet::builder()
        .literal("a".repeat(16))
        .literal("b".repeat(16))
        .build()
        .unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::MultiMemchr { .. }),
        "Expected MultiMemchr for exactly 16-byte literals"
    );

    let data = format!("{} {}", "a".repeat(16), "b".repeat(16));
    let expected = reference_scan(&patterns, data.as_bytes());
    let actual = patterns.scan(data.as_bytes()).unwrap();
    assert_eq!(expected, actual);
}

/// Verify SingleRegex strategy produces identical results to the default path.
#[test]
fn single_regex_parity_with_default() {
    let patterns = PatternSet::builder().regex(r"[a-z]+").build().unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::SingleRegex { .. }),
        "Expected SingleRegex strategy"
    );

    let test_cases = [
        b"".as_slice(),
        b"123456",      // No letters
        b"abc",         // Simple match
        b"123abc",      // Number then letters
        b"abc123",      // Letters then number
        b"abc 123 def", // Multiple matches
        b"ABC",         // Case sensitive (no match for lowercase pattern)
    ];

    for data in &test_cases {
        let expected = reference_scan(&patterns, data);
        let actual = patterns.scan(data).unwrap();
        assert_eq!(
            expected,
            actual,
            "Mismatch for input: {:?}",
            String::from_utf8_lossy(data)
        );
    }
}

/// Verify complex regex patterns work correctly.
#[test]
fn single_regex_complex_patterns() {
    let test_cases: [(&str, &[u8]); 4] = [
        (r"[0-9]+", b"abc 123 def 456"),
        (r"[a-z]+@[a-z]+\.[a-z]+", b"contact john@example.com now"),
        (r"https?://[^\s]+", b"visit https://example.com/path"),
        (r"test[0-9]+", b"a test123 here"),
    ];

    for (pattern, data) in &test_cases {
        let patterns = PatternSet::builder().regex(pattern).build().unwrap();
        let expected = reference_scan(&patterns, data);
        let actual = patterns.scan(data).unwrap();
        assert_eq!(
            expected,
            actual,
            "Mismatch for pattern {} on input: {:?}",
            pattern,
            String::from_utf8_lossy(data)
        );
    }
}

/// Verify FullDfa strategy (mixed patterns) produces identical results.
#[test]
fn full_dfa_mixed_patterns_parity() {
    let patterns = PatternSet::builder()
        .literal("needle")
        .regex(r"[0-9]+")
        .build()
        .unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::FullDfa),
        "Expected FullDfa for mixed patterns"
    );

    let test_cases = [
        b"no matches here".as_slice(),
        b"just a needle",
        b"just 123 numbers",
        b"needle with 123 numbers",
        b"123 needle 456",
        b"multiple needle needle and 123 456",
    ];

    for data in &test_cases {
        let expected = reference_scan(&patterns, data);
        let actual = patterns.scan(data).unwrap();
        assert_eq!(
            expected,
            actual,
            "Mismatch for input: {:?}",
            String::from_utf8_lossy(data)
        );
    }
}

/// Verify AhoCorasick strategy (many literals) produces identical results.
#[test]
fn aho_corasick_many_literals_parity() {
    let mut builder = PatternSet::builder();
    for i in 0..20 {
        builder = builder.literal(format!("pattern{:02}", i));
    }
    let patterns = builder.build().unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::AhoCorasick),
        "Expected AhoCorasick for many literals"
    );

    let data = b"pattern00 middle pattern05 pattern19";
    let expected = reference_scan(&patterns, data);
    let actual = patterns.scan(data).unwrap();
    assert_eq!(expected, actual);
}

/// Verify case-insensitive literals use Aho-Corasick (not memchr).
#[test]
fn case_insensitive_uses_aho_corasick_parity() {
    let patterns = PatternSet::builder()
        .literal("needle")
        .case_insensitive(true)
        .build()
        .unwrap();

    let strategy = ScanStrategy::select(&patterns);
    assert!(
        matches!(strategy, ScanStrategy::AhoCorasick),
        "Expected AhoCorasick for case-insensitive"
    );

    let test_cases = [
        (b"needle".as_slice(), true),
        (b"NEEDLE", true),
        (b"Needle", true),
        (b"nEeDlE", true),
        (b"noodle", false),
    ];

    for (data, should_match) in &test_cases {
        let matches = patterns.scan(data).unwrap();
        let found = !matches.is_empty();
        assert_eq!(
            found,
            *should_match,
            "Case-insensitive match failed for: {:?}",
            String::from_utf8_lossy(data)
        );
    }
}

/// Verify scan_count produces identical results.
#[test]
fn scan_count_parity() {
    let patterns = PatternSet::builder()
        .literal("foo")
        .literal("bar")
        .build()
        .unwrap();

    let data = b"foo bar foo baz bar foo";
    let expected_count = reference_scan(&patterns, data).len();
    let actual_count = patterns.scan_count(data).unwrap();
    assert_eq!(expected_count, actual_count);
}

/// Verify scan_with produces identical results.
#[test]
fn scan_with_parity() {
    let patterns = PatternSet::builder()
        .literal("foo")
        .literal("bar")
        .build()
        .unwrap();

    let data = b"foo bar foo baz bar";

    let mut expected = Vec::new();
    warpstate::cpu::scan_with(patterns.ir(), data, &mut |m| {
        expected.push(m);
        true
    })
    .unwrap();

    let mut actual = Vec::new();
    patterns
        .scan_with(data, |m| {
            actual.push(m);
            true
        })
        .unwrap();

    assert_eq!(expected, actual);
}

/// Verify scan_with early termination works correctly.
#[test]
fn scan_with_early_termination_parity() {
    let patterns = PatternSet::builder()
        .literal("foo")
        .literal("bar")
        .build()
        .unwrap();

    let data = b"foo bar foo baz bar foo";

    // Collect only first 2 matches
    let mut expected = Vec::new();
    let mut count = 0;
    warpstate::cpu::scan_with(patterns.ir(), data, &mut |m| {
        expected.push(m);
        count += 1;
        count < 2
    })
    .unwrap();

    let mut actual = Vec::new();
    count = 0;
    patterns
        .scan_with(data, |m| {
            actual.push(m);
            count += 1;
            count < 2
        })
        .unwrap();

    assert_eq!(expected.len(), 2);
    assert_eq!(expected, actual);
}

/// Test with binary data (non-UTF8).
#[test]
fn binary_data_parity() {
    let patterns = PatternSet::builder()
        .literal_bytes(vec![0x00, 0xFF, 0x42])
        .literal_bytes(vec![0x13, 0x37])
        .build()
        .unwrap();

    let data = vec![
        0x00, 0xFF, 0x42, // Match 1
        0x01, 0x02, // Gap
        0x13, 0x37, // Match 2
        0x00, 0xFF, 0x42, // Match 3
    ];

    let expected = reference_scan(&patterns, &data);
    let actual = patterns.scan(&data).unwrap();
    assert_eq!(expected, actual);
}

/// Test overlapping match semantics don't affect non-overlapping results.
#[test]
fn non_overlapping_semantics_preserved() {
    // Pattern "aa" on "aaaa" should give 2 non-overlapping matches: [0,2) and [2,4)
    let patterns = PatternSet::builder().literal("aa").build().unwrap();

    let data = b"aaaa";
    let expected = reference_scan(&patterns, data);
    let actual = patterns.scan(data).unwrap();

    assert_eq!(expected.len(), 2, "Expected 2 non-overlapping matches");
    assert_eq!(actual.len(), 2);
    assert_eq!(expected, actual);
}

/// Test pattern IDs are preserved correctly.
#[test]
fn pattern_ids_preserved() {
    let patterns = PatternSet::builder()
        .literal("first") // ID 0
        .literal("second") // ID 1
        .literal("third") // ID 2
        .build()
        .unwrap();

    let data = b"first second third first";
    let matches = patterns.scan(data).unwrap();

    // Find match at position 0 (should be "first", ID 0)
    let first_match = matches.iter().find(|m| m.start == 0).unwrap();
    assert_eq!(first_match.pattern_id, 0);

    // Find match at position 6 (should be "second", ID 1)
    let second_match = matches.iter().find(|m| m.start == 6).unwrap();
    assert_eq!(second_match.pattern_id, 1);

    // Find match at position 13 (should be "third", ID 2)
    let third_match = matches.iter().find(|m| m.start == 13).unwrap();
    assert_eq!(third_match.pattern_id, 2);
}

/// Test large input handling.
#[test]
fn large_input_parity() {
    let patterns = PatternSet::builder().literal("target").build().unwrap();

    // Generate ~100KB of data
    let mut data = Vec::new();
    for i in 0..1000 {
        if i % 100 == 0 {
            data.extend_from_slice(b"target");
        } else {
            data.extend_from_slice(b"filler data goes here with some content ");
        }
    }

    let expected = reference_scan(&patterns, &data);
    let actual = patterns.scan(&data).unwrap();
    assert_eq!(expected, actual);
    assert_eq!(actual.len(), 10); // Should find 10 matches
}

/// Test special regex characters in literal patterns.
#[test]
fn special_chars_in_literals_parity() {
    let patterns = PatternSet::builder()
        .literal("a.b")
        .literal("c*d")
        .literal("e[f]")
        .build()
        .unwrap();

    let data = b"a.b c*d e[f] x.y z*w";
    let expected = reference_scan(&patterns, data);
    let actual = patterns.scan(data).unwrap();
    assert_eq!(expected, actual);
}

/// Stress test with many matches.
#[test]
fn many_matches_parity() {
    let patterns = PatternSet::builder().literal("x").build().unwrap();

    // Create data with many 'x' characters
    let data = vec![b'x'; 1000];

    let expected = reference_scan(&patterns, &data);
    let actual = patterns.scan(&data).unwrap();
    assert_eq!(expected, actual);
    assert_eq!(actual.len(), 1000); // Should find 1000 matches
}

/// Verify empty pattern set fails to build (not a specialization issue but ensures API consistency).
#[test]
fn empty_pattern_set_fails() {
    let result = PatternSet::builder().build();
    assert!(result.is_err());
}

/// Test pattern names are preserved.
#[test]
fn pattern_names_preserved() {
    let patterns = PatternSet::builder()
        .named_literal("foo_name", "foo")
        .named_literal("bar_name", "bar")
        .build()
        .unwrap();

    let ir = patterns.ir();
    assert_eq!(ir.names[0], Some("foo_name".to_string()));
    assert_eq!(ir.names[1], Some("bar_name".to_string()));

    // Scan should still work correctly
    let data = b"foo bar foo";
    let expected = reference_scan(&patterns, data);
    let actual = patterns.scan(data).unwrap();
    assert_eq!(expected, actual);
}
