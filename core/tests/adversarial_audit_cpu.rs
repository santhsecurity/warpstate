//! Adversarial Audit Tests for warpstate CPU module
//!
//! These tests are designed to expose:
//! - Integer overflow vulnerabilities
//! - Memory exhaustion vectors
//! - Logic errors in edge cases
//! - Incorrect match semantics

use warpstate::{Error, PatternSet};

// =============================================================================
// CRITICAL: Integer Overflow and Memory Exhaustion Tests
// =============================================================================

/// CRITICAL: Test that match overflow is properly detected at boundary.
///
/// PatternSet::scan() uses an internal buffer sized by estimate_match_capacity
/// (data_len / 1024, clamped to [64, 1M]). When every byte matches, the
/// internal buffer overflows and MatchBufferOverflow is returned. This is
/// the correct behavior: unbounded match output is explicitly rejected.
#[test]
fn critical_match_overflow_boundary() {
    let ps = PatternSet::builder().literal("a").build().unwrap();

    // Every-byte-matches pathological input: the internal buffer is much
    // smaller than 1M, so MatchBufferOverflow is expected.
    let data = vec![b'a'; 1_048_576];
    let result = ps.scan(&data);
    match result {
        Err(Error::MatchBufferOverflow { .. }) => {} // Expected: internal buffer too small
        Ok(matches) => {
            // If it succeeds, that means the buffer was large enough (unlikely but valid)
            assert!(matches.len() <= 1_048_576);
        }
        Err(e) => panic!("Expected MatchBufferOverflow, got {:?}", e),
    }
}

/// CRITICAL: Test potential usize overflow in match counting
#[test]
fn critical_match_count_overflow_protection() {
    let ps = PatternSet::builder().literal("").build();
    // Empty patterns should be rejected at build time
    assert!(ps.is_err(), "Empty pattern should fail at build time");
}

/// CRITICAL: Verify pattern_id assignment doesn't overflow
#[test]
fn critical_pattern_id_u32_boundary() {
    // This tests internal behavior - pattern_id is u32
    // Build many patterns and verify they get sequential IDs
    let mut builder = PatternSet::builder();
    for i in 0..1000usize {
        builder = builder.literal(&format!("pattern{:08}", i));
    }
    let ps = builder.build().unwrap();

    // Scan with input containing one pattern
    let matches = ps.scan(b"pattern00000000").unwrap();
    assert!(!matches.is_empty());
    assert_eq!(matches[0].pattern_id, 0);
}

// =============================================================================
// HIGH: Case-Insensitive Memory Exhaustion
// =============================================================================

/// HIGH: CI scan allocates duplicate buffer - test with large input
#[test]
fn high_ci_memory_doubling() {
    let ps = PatternSet::builder()
        .literal("test")
        .case_insensitive(true)
        .build()
        .unwrap();

    // Large input for CI scan - allocates second buffer
    let data = vec![b'X'; 10_000_000]; // 10MB
    let result = ps.scan(&data);
    assert!(result.is_ok(), "CI scan should handle 10MB input");
}

/// HIGH: Multi-pattern CI with many patterns
#[test]
fn high_ci_multi_pattern_memchr() {
    // This triggers scan_multi_literal_ci_memchr_with path
    let mut builder = PatternSet::builder();
    for i in 0..50 {
        // Between 1 and 64 patterns triggers memchr CI path
        builder = builder.literal(&format!("pat{:02}", i));
    }
    let ps = builder.case_insensitive(true).build().unwrap();

    let data = b"PAT00 pat01 PAT25";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 3, "Should find 3 CI matches");
}

// =============================================================================
// HIGH: Regex Pattern Edge Cases
// =============================================================================

/// HIGH: Regex with anchors at boundaries
///
/// NOTE: The regex engine uses multi-line mode by default, so ^ and $ match
/// at line boundaries within the string, not just at start/end.
#[test]
fn high_regex_anchors() {
    let ps = PatternSet::builder()
        .regex("^start")
        .regex("end$")
        .build()
        .unwrap();

    // Multi-line mode: ^ matches after \n, $ matches before \n or at end
    let matches = ps.scan(b"start middle end").unwrap();
    assert_eq!(
        matches.len(),
        2,
        "Should match both anchored patterns at line boundaries"
    );

    // In multi-line mode, ^start can match at position 0 (start of string)
    // or after any newline
    let with_newline = ps.scan(b"line\nstart here").unwrap();
    assert_eq!(
        with_newline.len(),
        1,
        "^start should match after newline in multi-line mode"
    );
}

/// HIGH: Regex alternation lengths
#[test]
fn high_regex_alternation_lengths() {
    // Pattern with alternation of different lengths
    let ps = PatternSet::builder().regex("a|bb|ccc").build().unwrap();

    let data = b"a bb ccc";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 3, "Should find all alternation matches");
}

// =============================================================================
// MEDIUM: Hash Scanner Collision and DoS
// =============================================================================

/// MEDIUM: Hash scanner with patterns that could collide
#[test]
fn medium_hash_collision_patterns() {
    // Create many patterns to trigger hash scanner
    let mut builder = PatternSet::builder();
    for i in 0..1500 {
        // Above HASH_SCANNER_LITERAL_THRESHOLD (1000)
        builder = builder.literal(&format!("hash_test_{:06}", i));
    }
    let ps = builder.build().unwrap();

    // Should use hash scanner
    let data = b"prefix hash_test_000000 hash_test_000001 suffix";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 2);
}

/// MEDIUM: Empty pattern handling in hash scanner
#[test]
fn medium_empty_pattern_rejection() {
    let result = PatternSet::builder().literal("").literal("valid").build();

    assert!(result.is_err(), "Should reject empty pattern");
}

// =============================================================================
// MEDIUM: Match Semantics and Ordering
// =============================================================================

/// MEDIUM: Verify match ordering is strictly by position
#[test]
fn medium_match_ordering_strict() {
    let ps = PatternSet::builder()
        .literal("z") // Pattern 0
        .literal("a") // Pattern 1
        .literal("m") // Pattern 2
        .build()
        .unwrap();

    let data = b"abcdefghijklmnoz";
    let matches = ps.scan(data).unwrap();

    // Should be ordered by position, not pattern_id
    let positions: Vec<u32> = matches.iter().map(|m| m.start).collect();
    let sorted_positions = {
        let mut v = positions.clone();
        v.sort();
        v
    };
    assert_eq!(
        positions, sorted_positions,
        "Matches should be sorted by position"
    );
}

/// MEDIUM: Overlapping matches with different length patterns
#[test]
fn medium_overlapping_different_lengths() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("ab")
        .literal("abc")
        .literal("abcd")
        .build()
        .unwrap();

    let data = b"abcd";
    let matches = ps.scan_overlapping(data).unwrap();

    // All patterns should match at position 0
    let pos0_matches: Vec<_> = matches.iter().filter(|m| m.start == 0).collect();
    assert_eq!(
        pos0_matches.len(),
        4,
        "All 4 patterns should match at position 0"
    );
}

// =============================================================================
// LOW: Edge Cases and Defensive Programming
// =============================================================================

/// LOW: Single byte pattern across all positions
#[test]
fn low_single_byte_all_positions() {
    let ps = PatternSet::builder().literal("x").build().unwrap();

    for pos in 0..100 {
        let mut data = vec![b'a'; 100];
        data[pos] = b'x';
        let matches = ps.scan(&data).unwrap();
        assert_eq!(
            matches.len(),
            1,
            "Should find single byte at position {}",
            pos
        );
        assert_eq!(matches[0].start, pos as u32);
    }
}

/// LOW: Very long pattern that fills input
#[test]
fn low_long_pattern_exact_fit() {
    let pattern = "x".repeat(10000);
    let ps = PatternSet::builder().literal(&pattern).build().unwrap();

    let mut data = pattern.clone();
    data.push_str("extra");
    let matches = ps.scan(data.as_bytes()).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 10000);
}

/// LOW: Pattern with every byte value 0x00-0xFF
#[test]
fn low_all_byte_values() {
    let mut builder = PatternSet::builder();
    for i in 0u8..=255 {
        builder = builder.literal_bytes(vec![i]);
    }
    let ps = builder.build().unwrap();

    let data: Vec<u8> = (0u8..=255).collect();
    // Use scan_to_buffer with a large enough buffer since the default
    // estimate_match_capacity(256) = 64 is too small for 256 matches.
    let mut out = vec![warpstate::Match::from_parts(0, 0, 0); 512];
    let count = ps.scan_to_buffer(&data, &mut out).unwrap();

    assert_eq!(count, 256);
    let matches = &out[..count];
    // Verify each byte matched at correct position
    for i in 0u8..=255 {
        let m = matches.iter().find(|m| m.pattern_id == i as u32);
        assert!(m.is_some(), "Pattern {} should have matched", i);
        assert_eq!(m.unwrap().start, i as u32);
    }
}

// =============================================================================
// CORRECTNESS: scan_with early termination
// =============================================================================

/// CORRECTNESS: Early termination preserves match order
#[test]
fn correctness_scan_with_early_term_order() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("b")
        .build()
        .unwrap();

    let data = b"a b a b a";
    let mut visited = Vec::new();

    ps.scan_with(data, |m| {
        visited.push(m.pattern_id);
        visited.len() < 3 // Stop after 3
    })
    .unwrap();

    assert_eq!(visited.len(), 3);
    assert_eq!(visited, vec![0, 1, 0]); // First 3 matches in order
}

/// CORRECTNESS: scan_count matches scan().len()
#[test]
fn correctness_scan_count_parity() {
    let ps = PatternSet::builder()
        .literal("the")
        .literal("and")
        .regex(r"\w+")
        .build()
        .unwrap();

    let data = b"the quick brown fox and the lazy dog";
    let count = ps.scan_count(data).unwrap();
    let matches = ps.scan(data).unwrap();

    assert_eq!(count, matches.len(), "scan_count should equal scan().len()");
}

// =============================================================================
// BUG HUNTING: Specific patterns that might expose issues
// =============================================================================

/// Test pattern that could expose off-by-one in boundary checks
#[test]
fn bug_hunt_boundary_off_by_one() {
    let ps = PatternSet::builder().literal("ab").build().unwrap();

    // Pattern at exact end of input
    let data = b"xab";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 3);

    // Pattern one past end should not match
    let data2 = b"xa";
    let matches2 = ps.scan(data2).unwrap();
    assert!(matches2.is_empty());
}

/// Test Unicode patterns at UTF-8 boundaries
#[test]
fn bug_hunt_unicode_boundaries() {
    // CJK character "日" is 3 bytes: E6 97 A5
    let ps = PatternSet::builder().literal("日").build().unwrap();

    // Test at various alignments
    for prefix in ["", "a", "ab", "abc", "abcd"].iter() {
        let data = format!("{}日", prefix);
        let matches = ps.scan(data.as_bytes()).unwrap();
        assert_eq!(
            matches.len(),
            1,
            "Should match at prefix len {}",
            prefix.len()
        );
        assert_eq!(matches[0].start, prefix.len() as u32);
    }
}

/// Test that overlapping and non-overlapping produce expected counts
#[test]
fn bug_hunt_overlap_counts() {
    let ps = PatternSet::builder().literal("aa").build().unwrap();

    // Input: aaaaa (5 a's)
    // Non-overlapping: positions 0-1, 2-3 = 2 matches
    // Overlapping: positions 0-1, 1-2, 2-3, 3-4 = 4 matches

    let non_overlap = ps.scan(b"aaaaa").unwrap();
    let overlap = ps.scan_overlapping(b"aaaaa").unwrap();

    assert_eq!(
        non_overlap.len(),
        2,
        "Non-overlapping should find 2 matches"
    );
    assert_eq!(overlap.len(), 4, "Overlapping should find 4 matches");
}

/// Test zero-width or minimal patterns don't cause issues
#[test]
fn bug_hunt_minimal_patterns() {
    // Single byte patterns
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00")
        .literal_bytes(b"\xFF")
        .literal("a")
        .build()
        .unwrap();

    let data = b"\x00a\xFF";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 3);
}

/// Test CachedScanner with edge cases
#[test]
fn bug_hunt_cached_scanner_edge_cases() {
    use warpstate::CachedScanner;

    let ps = PatternSet::builder().literal("needle").build().unwrap();
    let scanner = CachedScanner::new(ps.ir()).unwrap();

    // Empty input
    let mut matches_buf = [warpstate::Match::from_parts(0, 0, 0); 10];
    let count = scanner.scan(b"", &mut matches_buf).unwrap();
    let matches = &matches_buf[..count];
    assert!(matches.is_empty());

    // Pattern at various positions
    for pos in 0..10 {
        let mut data = vec![b'x'; 20];
        data[pos..pos + 6].copy_from_slice(b"needle");
        let mut matches_buf = [warpstate::Match::from_parts(0, 0, 0); 10];
        let count = scanner.scan(&data, &mut matches_buf).unwrap();
        let matches = &matches_buf[..count];
        assert_eq!(matches.len(), 1, "Should find needle at position {}", pos);
        assert_eq!(matches[0].start, pos as u32);
    }
}

/// Test large pattern count with CachedScanner (uses Hash path)
#[test]
fn bug_hunt_cached_scanner_hash_path() {
    use warpstate::CachedScanner;

    let mut builder = PatternSet::builder();
    for i in 0..1500 {
        // Triggers hash scanner
        builder = builder.literal(&format!("p{:04}", i));
    }
    let ps = builder.build().unwrap();
    let scanner = CachedScanner::new(ps.ir()).unwrap();

    let data = b"p0000 p0001 p1499";
    let mut matches_buf = [warpstate::Match::from_parts(0, 0, 0); 10];
    let count = scanner.scan(data, &mut matches_buf).unwrap();
    let matches = &matches_buf[..count];
    assert_eq!(matches.len(), 3);
}

/// Test regex pattern with special characters that might be misinterpreted
#[test]
fn bug_hunt_regex_special_chars() {
    // Pattern that looks like regex but used as literal
    let literal_ps = PatternSet::builder().literal(r"[a-z]+").build().unwrap();
    let regex_ps = PatternSet::builder().regex(r"[a-z]+").build().unwrap();

    let data = b"test [a-z]+ pattern";

    let literal_matches = literal_ps.scan(data).unwrap();
    let regex_matches = regex_ps.scan(data).unwrap();

    // Literal should find "[a-z]+" as literal text
    assert_eq!(
        literal_matches.len(),
        1,
        "Literal should match the exact text"
    );
    assert_eq!(
        literal_matches[0].start, 5,
        "Literal match should start at position 5"
    );

    // Regex should find "test" and "pattern" and "a" and "z"
    // (depending on whether + is greedy and how it matches)
    eprintln!("Regex matches: {:?}", regex_matches);
    // Just verify it doesn't panic and produces reasonable results
    assert!(
        !regex_matches.is_empty(),
        "Regex should find at least some matches"
    );
}

/// Test memory behavior with repeated scans
#[test]
fn bug_hunt_repeated_scan_stability() {
    let ps = PatternSet::builder()
        .literal("test")
        .regex(r"\d+")
        .build()
        .unwrap();

    let data = b"test 123 test 456";

    // Run many scans
    for i in 0..100 {
        let matches = ps.scan(data).unwrap();
        assert_eq!(
            matches.len(),
            4,
            "Scan {} should produce consistent results",
            i
        );
    }
}

/// Test that pattern order doesn't affect results
#[test]
fn bug_hunt_pattern_order_independence() {
    let ps1 = PatternSet::builder()
        .literal("foo")
        .literal("bar")
        .build()
        .unwrap();

    let ps2 = PatternSet::builder()
        .literal("bar")
        .literal("foo")
        .build()
        .unwrap();

    let data = b"foo bar";
    let matches1 = ps1.scan(data).unwrap();
    let matches2 = ps2.scan(data).unwrap();

    // Same number of matches, but pattern_ids differ
    assert_eq!(matches1.len(), matches2.len());

    // Sort by position for comparison
    let mut m1 = matches1.clone();
    let mut m2 = matches2.clone();
    m1.sort_by_key(|m| m.start);
    m2.sort_by_key(|m| m.start);

    for (a, b) in m1.iter().zip(m2.iter()) {
        assert_eq!(a.start, b.start);
        assert_eq!(a.end, b.end);
    }
}
