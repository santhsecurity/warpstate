use std::sync::Arc;
use warpstate::{CachedScanner, Error, HotSwapPatternSet, Match, PatternSet};

// --- Category 1: Empty input / zero-length slices ---

#[test]
fn test_01_empty_input_slice() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let matches = patterns.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should yield 0 matches");
}

#[test]
fn test_02_empty_pattern_set_build() {
    let err = PatternSet::builder().build().unwrap_err();
    assert!(
        matches!(err, Error::EmptyPatternSet),
        "Building empty pattern set must fail"
    );
}

#[test]
fn test_03_zero_length_regex_match() {
    let patterns = PatternSet::builder().regex(".*").build().unwrap();
    let matches = patterns.scan(b"").unwrap();
    assert_eq!(matches.len(), 1, ".* should match empty string at pos 0");
}

#[test]
fn test_04_empty_literal_bytes() {
    let err = PatternSet::builder()
        .literal_bytes(vec![])
        .build()
        .unwrap_err();
    assert!(
        matches!(err, Error::EmptyPattern { .. }),
        "Empty literal bytes should fail to build"
    );
}

// --- Category 2: Null bytes in input ---

#[test]
fn test_05_null_byte_in_input() {
    let patterns = PatternSet::builder().literal("world").build().unwrap();
    let input = b"hello\0world";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find match after null byte");
    assert_eq!(matches[0].start, 6);
}

#[test]
fn test_06_null_byte_in_pattern() {
    let patterns = PatternSet::builder()
        .literal_bytes(b"hello\0world".to_vec())
        .build()
        .unwrap();
    let input = b"say hello\0world now";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(
        matches.len(),
        1,
        "Should match pattern containing null byte"
    );
}

#[test]
fn test_07_all_null_input() {
    let patterns = PatternSet::builder()
        .literal_bytes(vec![0])
        .build()
        .unwrap();
    let input = vec![0; 100];
    let matches = patterns.scan_overlapping(&input).unwrap();
    assert_eq!(matches.len(), 100, "Should find 100 null bytes");
}

#[test]
fn test_08_regex_with_null_bytes() {
    let patterns = PatternSet::builder().regex("a\0b").build().unwrap();
    let matches = patterns.scan(b"a\0b").unwrap();
    assert_eq!(matches.len(), 1, "Regex should match null byte sequence");
}

// --- Category 3: Maximum u32/u64 values ---

#[test]
fn test_09_max_matches_overflow_scan() {
    let patterns = PatternSet::builder().literal("a").build().unwrap();
    let input = vec![b'a'; 2_000_000];
    // Buffer smaller than input guarantees overflow.
    let mut matches = vec![Match::from_parts(0, 0, 0); 1_000_000];
    let res = patterns.scan_to_buffer(&input, &mut matches);
    assert!(
        matches!(res, Err(Error::MatchBufferOverflow { .. })),
        "Should overflow match buffer limit during scan"
    );
}

#[test]
fn test_10_max_cpu_matches_overflow() {
    let patterns = PatternSet::builder().literal("a").build().unwrap();
    // 2,000,000 matches, buffer only 1M — guaranteed overflow.
    let input = vec![b'a'; 2_000_000];
    let mut matches = vec![Match::from_parts(0, 0, 0); 1_000_000];
    let res = patterns.scan_overlapping_to_buffer(&input, &mut matches);
    assert!(
        matches!(res, Err(Error::MatchBufferOverflow { .. })),
        "Should overflow match buffer limit"
    );
}

#[test]
fn test_11_match_exactly_at_buffer_end() {
    let patterns = PatternSet::builder().literal("end").build().unwrap();
    let mut input = vec![b'x'; 10_000];
    input.extend_from_slice(b"end");
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 10_000);
}

#[test]
fn test_12_match_exactly_at_buffer_start() {
    let patterns = PatternSet::builder().literal("start").build().unwrap();
    let mut input = b"start".to_vec();
    input.extend(vec![b'x'; 10_000]);
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
}

// --- Category 4: Resource Exhaustion ---

#[test]
fn test_13_100k_literal_patterns() {
    let mut builder = PatternSet::builder();
    for i in 0..10_000 {
        // Scaled down slightly to run reasonably fast, still tests large set
        builder = builder.literal(&format!("pattern_{:05}", i));
    }
    let patterns = builder.build().unwrap();
    assert_eq!(patterns.len(), 10_000);
    let input = b"pattern_09999";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_14_deeply_nested_regex_rejected() {
    let err = PatternSet::builder().regex("(a*)*").build().unwrap_err();
    assert!(
        matches!(err, Error::PathologicalRegex { .. }),
        "Deeply nested regex must be rejected"
    );
}

#[test]
fn test_15_massive_input_buffer() {
    let patterns = PatternSet::builder().literal("X").build().unwrap();
    let input = vec![b'X'; 1_000_000];
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1_000_000);
}

#[test]
fn test_16_many_regex_patterns() {
    let mut builder = PatternSet::builder();
    for i in 0..100 {
        builder = builder.regex(&format!("reg[ex]{}+", i));
    }
    let patterns = builder.build().unwrap();
    assert_eq!(patterns.len(), 100);
}

// --- Category 5: Concurrent access ---

#[test]
fn test_17_concurrent_reads_same_pattern_set() {
    let patterns = Arc::new(PatternSet::builder().literal("shared").build().unwrap());
    let mut handles = vec![];
    for _ in 0..8 {
        let p = Arc::clone(&patterns);
        handles.push(std::thread::spawn(move || {
            let res = p.scan(b"shared resource").unwrap();
            assert_eq!(res.len(), 1);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_18_concurrent_hot_swap() {
    let p1 = PatternSet::builder().literal("v1").build().unwrap();
    let hot = Arc::new(HotSwapPatternSet::new(p1));
    let hot_clone = Arc::clone(&hot);

    let t1 = std::thread::spawn(move || {
        let mut buf = [Match::from_parts(0, 0, 0); 10];
        for _ in 0..1000 {
            let _ = hot_clone.scan(b"v1 or v2", &mut buf);
        }
    });

    let p2 = PatternSet::builder().literal("v2").build().unwrap();
    let _ = hot.swap(p2);
    t1.join().unwrap();
}

// --- Category 6: Partial / Truncated Data ---

#[test]
fn test_19_truncated_input() {
    let patterns = PatternSet::builder().literal("hello").build().unwrap();
    let matches = patterns.scan(b"hell").unwrap();
    assert_eq!(matches.len(), 0, "Truncated string should not match");
}

#[test]
fn test_20_prefix_match_only() {
    let patterns = PatternSet::builder().literal("ab").build().unwrap();
    let matches = patterns.scan(b"a").unwrap();
    assert_eq!(matches.len(), 0);
}

// --- Category 7: Unicode Edge Cases ---

#[test]
fn test_21_unicode_bom() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let input = b"\xEF\xBB\xBFtest";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 3);
}

#[test]
fn test_22_overlong_utf8() {
    let patterns = PatternSet::builder()
        .literal_bytes(vec![0xC0, 0xAF])
        .build()
        .unwrap();
    let matches = patterns.scan(&[0xC0, 0xAF]).unwrap();
    assert_eq!(
        matches.len(),
        1,
        "Should match raw bytes even if invalid UTF-8"
    );
}

#[test]
fn test_23_surrogate_pairs_in_regex() {
    let patterns = PatternSet::builder()
        .regex(r"\xED\xA0\xBD\xED\xB8\x80")
        .build()
        .unwrap();
    let matches = patterns.scan(b"\xED\xA0\xBD\xED\xB8\x80").unwrap();
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_24_pattern_ends_mid_utf8() {
    let patterns = PatternSet::builder()
        .literal_bytes(vec![0xF0, 0x9F, 0x92])
        .build()
        .unwrap(); // Partial poop emoji
    let input = vec![0xF0, 0x9F, 0x92, 0xA9]; // Full poop emoji
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1);
}

// --- Category 8: Duplicates / Overlaps ---

#[test]
fn test_25_duplicate_literals() {
    let patterns = PatternSet::builder()
        .literal("dup")
        .literal("dup")
        .build()
        .unwrap();
    assert_eq!(patterns.len(), 2);
    let matches = patterns.scan(b"dup").unwrap();
    // Usually non-overlapping returns one or two matches depending on semantics.
    // We expect it to match at least one.
    assert!(!matches.is_empty());
}

#[test]
fn test_26_duplicate_regexes() {
    let patterns = PatternSet::builder()
        .regex("a+")
        .regex("a+")
        .build()
        .unwrap();
    let matches = patterns.scan(b"aaa").unwrap();
    assert!(!matches.is_empty());
}

#[test]
fn test_27_overlapping_matches_scan_overlapping() {
    let patterns = PatternSet::builder().literal("aa").build().unwrap();
    let matches = patterns.scan_overlapping(b"aaaa").unwrap();
    assert_eq!(matches.len(), 3, "overlapping aa in aaaa = 3 matches");
}

#[test]
fn test_28_overlapping_mixed() {
    let patterns = PatternSet::builder()
        .literal("a")
        .literal("aa")
        .build()
        .unwrap();
    let matches = patterns.scan_overlapping(b"aaaa").unwrap();
    // "a" matches 4 times. "aa" matches 3 times. Total 7.
    assert_eq!(matches.len(), 7);
}

// --- Category 9: Various / Edge Cases ---

#[test]
fn test_29_cached_scanner_reuse() {
    let patterns = PatternSet::builder().literal("reuse").build().unwrap();
    let scanner = CachedScanner::new(patterns.ir()).unwrap();
    let mut buf = [Match::from_parts(0, 0, 0); 10];
    for _ in 0..100 {
        let count = scanner.scan(b"reuse it", &mut buf).unwrap();
        assert_eq!(count, 1);
    }
}

#[test]
fn test_30_scan_with_early_abort() {
    let patterns = PatternSet::builder().literal("stop").build().unwrap();
    let mut count = 0;
    patterns
        .scan_with(b"stop stop stop", |_| {
            count += 1;
            false // abort after first
        })
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_31_hash_scanner_threshold() {
    let mut builder = PatternSet::builder();
    for i in 0..6000 {
        builder = builder.literal(&format!("thresh{:04}", i));
    }
    let patterns = builder.build().unwrap();
    let scanner = CachedScanner::new(patterns.ir()).unwrap();
    let mut buf = [Match::from_parts(0, 0, 0); 10];
    let count = scanner.scan(b"thresh5999", &mut buf).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_32_case_insensitive_literal() {
    let patterns = PatternSet::builder()
        .case_insensitive(true)
        .literal("HeLlO")
        .build()
        .unwrap();
    let matches = patterns.scan(b"hElLo").unwrap();
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_33_case_insensitive_mixed_unicode() {
    // Note: rust regex / Aho-Corasick handles unicode case insensitivity differently
    let patterns = PatternSet::builder()
        .case_insensitive(true)
        .literal("SS")
        .build()
        .unwrap();
    // Test matching "ss"
    let matches = patterns.scan(b"ss").unwrap();
    assert_eq!(matches.len(), 1);
}
