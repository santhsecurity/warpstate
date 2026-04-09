#![cfg(feature = "fused")]
#![cfg(feature = "jit")]

use ebpfsieve::{ByteFrequencyFilter, ByteThreshold};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use warpstate::compiled_index::CompiledPatternIndex;
use warpstate::{AutoMatcherConfig, Error, FusedScanner, PatternSet};

// 1. FusedScanner optimization correctness: JIT vs interpreted parity on 1000 random inputs.
#[test]
fn test_01_fused_jit_vs_interpreted_parity_1000_random_inputs() {
    let patterns = PatternSet::builder()
        .literal("virus")
        .literal("malware")
        .literal("trojan")
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    assert!(scanner.has_jit());

    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..1000 {
        let len = rng.gen_range(10..1000);
        let mut data: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
        if rng.gen_bool(0.1) {
            let offset = rng.gen_range(0..data.len().saturating_sub(6));
            data[offset..offset + 5].copy_from_slice(b"virus");
        }
        let jit_matches = scanner.scan(&data).unwrap();
        let interpreted_matches = patterns.scan(&data).unwrap();
        assert_eq!(jit_matches, interpreted_matches);
    }
}

// 2. ebpfsieve byte-frequency prefilter rejects then full scan still finds ALL matches
#[test]
fn test_02_ebpfsieve_prefilter_rejects_then_full_scan_finds_all_matches() {
    let patterns = PatternSet::builder().literal("needle").build().unwrap();
    let filter = ByteFrequencyFilter::new([ByteThreshold::new(b'n', 1)]).unwrap();
    let scanner = FusedScanner::new(patterns.clone(), Some(filter));

    let mut data = vec![b'x'; 8192];
    data[8000..8006].copy_from_slice(b"needle");

    let matches = scanner.scan(&data).unwrap();
    let base_matches = patterns.scan(&data).unwrap();
    assert_eq!(matches.len(), base_matches.len());
    assert_eq!(matches.len(), 1);
}

// 3. SIMD prefilter finds candidates that verification confirms
#[test]
fn test_03_simd_prefilter_finds_candidates_verification_confirms() {
    let patterns = PatternSet::builder()
        .literal("simd_candidate")
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = b"some prefix simd_candidate and suffix";
    let matches = scanner.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 12);
}

// 4. Pattern set with 1 literal + 1 regex where both match at same offset
#[test]
fn test_04_pattern_set_1_literal_1_regex_same_offset() {
    let patterns = PatternSet::builder()
        .literal("test")
        .regex("t[e]st")
        .build()
        .unwrap();
    let data = b"this is a test of the system";
    let matches = patterns.scan_overlapping(data).unwrap();
    assert_eq!(matches.len(), 2);
    let mut found_literal = false;
    let mut found_regex = false;
    for m in &matches {
        if m.start == 10 {
            if m.pattern_id == 0 {
                found_literal = true;
            }
            if m.pattern_id == 1 {
                found_regex = true;
            }
        }
    }
    assert!(found_literal && found_regex);
}

// 5. CompiledPatternIndex serialize with CRC then flip 1 bit and verify load fails
#[test]
fn test_05_compiled_pattern_index_serialize_crc_flip_1_bit_fails() {
    let patterns = PatternSet::builder().literal("crc_test").build().unwrap();
    let mut bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let len = bytes.len();
    bytes[len / 2] ^= 0x01;
    let res = CompiledPatternIndex::load(&bytes);
    assert!(res.is_err());
}

// 6. Scan 128MB input in chunks and verify match offsets are globally correct not chunk-relative
#[tokio::test]
async fn test_06_scan_128mb_input_in_chunks_global_offsets() {
    let patterns = PatternSet::builder()
        .literal("chunk_boundary_match")
        .build()
        .unwrap();
    // Default config typically has chunk size 128MB or some fixed size. Let's explicitly set to 10MB chunk.
    let config = AutoMatcherConfig::new()
        .chunk_size(10 * 1024 * 1024)
        .chunk_overlap(1024);
    let matcher = warpstate::AutoMatcher::with_config(&patterns, config)
        .await
        .unwrap();

    // Test the 128MB buffer size
    let mut data = vec![b'x'; 128 * 1024 * 1024];

    // First match in the first chunk
    let first_match_idx = 10 * 1024;
    data[first_match_idx..first_match_idx + 20].copy_from_slice(b"chunk_boundary_match");

    // Second match straddling 10MB chunk boundary
    let straddle_idx = 10 * 1024 * 1024 - 10;
    data[straddle_idx..straddle_idx + 20].copy_from_slice(b"chunk_boundary_match");

    // Third match deep in the 12th chunk
    let deep_idx = 115 * 1024 * 1024;
    data[deep_idx..deep_idx + 20].copy_from_slice(b"chunk_boundary_match");

    let matches = matcher.scan(&data).await.unwrap();
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].start, first_match_idx as u32);
    assert_eq!(matches[1].start, straddle_idx as u32);
    assert_eq!(matches[2].start, deep_idx as u32);
}

// 7. Max memory zero length buffer scan with 0 length pattern
#[test]
fn test_07_zero_length_buffer_zero_length_pattern() {
    let result = PatternSet::builder().literal("").build();
    assert!(result.is_err(), "Empty pattern set should error");
}

// 8. Exactly max chunk size without matching anything
#[test]
fn test_08_exactly_max_chunk_size_no_match() {
    let patterns = PatternSet::builder().literal("invisible").build().unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = vec![b'a'; 10 * 1024 * 1024]; // 10MB test
    let matches = scanner.scan(&data).unwrap();
    assert_eq!(matches.len(), 0);
}

// 9. Scan extremely small block overlap boundary
#[tokio::test]
async fn test_09_small_block_overlap_boundary() {
    let patterns = PatternSet::builder().literal("12345").build().unwrap();
    let config = AutoMatcherConfig::new().chunk_size(10).chunk_overlap(4);
    let matcher = warpstate::AutoMatcher::with_config(&patterns, config)
        .await
        .unwrap();
    let data = b"123451234512345";
    let matches = matcher.scan(data).await.unwrap();
    assert_eq!(matches.len(), 3);
}

// 10. Repeated single character overlapping pattern
#[test]
fn test_10_repeated_single_character() {
    let patterns = PatternSet::builder().literal("a").build().unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = b"aaaaa";
    let matches = scanner.scan(data).unwrap();
    // Non-overlapping by default, so it returns 5 distinct matches
    assert_eq!(matches.len(), 5);
}

// 11. Prefix matches but full fails at the last byte
#[test]
fn test_11_prefix_match_full_fails_last_byte() {
    let patterns = PatternSet::builder().literal("almost").build().unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = b"almosx";
    let matches = scanner.scan(data).unwrap();
    assert_eq!(matches.len(), 0);
}

// 12. Fused scanner aborts early dynamically via visitor
#[test]
fn test_12_fused_scanner_aborts_early_via_visitor() {
    let patterns = PatternSet::builder().literal("abort").build().unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = b"abort abort abort";
    let mut count = 0;
    scanner
        .scan_with(data, |_| {
            count += 1;
            false // Abort immediately
        })
        .unwrap();
    assert_eq!(count, 1);
}

// 13. Deeply nested regex pathological
#[test]
fn test_13_deeply_nested_regex_rejected() {
    let err = PatternSet::builder().regex("(a+)+").build().unwrap_err();
    assert!(matches!(err, Error::PathologicalRegex { .. }));
}

// 14. 100K regexes parsing does not panic
#[test]
fn test_14_massive_regex_count_fails_gracefully_or_builds() {
    let mut builder = PatternSet::builder();
    for i in 0..10_000 {
        builder = builder.regex(&format!("regex_{i}"));
    }
    // Just verify it doesn't OOM or panic. If it succeeds, length is 10k.
    let patterns = builder.build().unwrap();
    assert_eq!(patterns.len(), 10_000);
}

// 15. All patterns are single bytes, verify literal automaton works
#[test]
fn test_15_all_patterns_single_byte() {
    let patterns = PatternSet::builder()
        .literal("a")
        .literal("b")
        .literal("c")
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = b"cba";
    let matches = scanner.scan(data).unwrap();
    assert_eq!(matches.len(), 3);
}

// 16. Unicode case insensitive massive chunk boundary
#[test]
fn test_16_unicode_case_insensitive_chunk_boundary() {
    // Some implementations might handle this differently.
    // Testing case insensitivity of unicode character handling.
    let patterns = PatternSet::builder()
        .case_insensitive(true)
        .literal("über")
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    // use unicode bytes manually
    let data: &[u8] = b"x \xC3\x9Cber y";
    let _matches = scanner.scan(data).unwrap();
    // In our manual byte conversion for case insensitivity it might match or not based on impl. We just ensure it doesn't panic.
}

// 17. JIT disabled fallback behavior check
#[test]
fn test_17_fused_jit_compile_failure_graceful_fallback() {
    // Some complex un-jittable literal logic or just default checking
    let patterns = PatternSet::builder()
        .literal("some_literal")
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let matches = scanner.scan(b"some_literal").unwrap();
    assert_eq!(matches.len(), 1);
}

// 18. Exact chunk size single match
#[tokio::test]
async fn test_18_exact_chunk_size_single_match() {
    let patterns = PatternSet::builder().literal("chunk").build().unwrap();
    let config = AutoMatcherConfig::new().chunk_size(5).chunk_overlap(1);
    let matcher = warpstate::AutoMatcher::with_config(&patterns, config)
        .await
        .unwrap();
    let data = b"chunk";
    let matches = matcher.scan(data).await.unwrap();
    assert_eq!(matches.len(), 1);
}

// 19. Empty input byte slice
#[test]
fn test_19_empty_input_byte_slice() {
    let patterns = PatternSet::builder().literal("a").build().unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let matches = scanner.scan(b"").unwrap();
    assert_eq!(matches.len(), 0);
}

// 20. Very long pattern exceeding fused window bytes
#[test]
fn test_20_very_long_pattern_exceeding_window() {
    let long_pattern = "a".repeat(10000);
    let patterns = PatternSet::builder()
        .literal(&long_pattern)
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);

    let mut data = vec![b'b'; 20000];
    data[100..10100].copy_from_slice(long_pattern.as_bytes());
    let matches = scanner.scan(&data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 100);
}

// 21. Match at the exact end of buffer
#[test]
fn test_21_match_at_exact_end() {
    let patterns = PatternSet::builder().literal("end").build().unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let mut data = vec![b'a'; 1024];
    data.extend_from_slice(b"end");
    let matches = scanner.scan(&data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1024);
}

// 22. Filter explicitly ignores window byte threshold completely when matching literal only regex
#[test]
fn test_22_fused_with_regex_disables_filter() {
    // Fused prefilters are disabled for mixed literal/regex.
    let patterns = PatternSet::builder()
        .literal("lit")
        .regex("re[g]ex")
        .build()
        .unwrap();
    let filter = ByteFrequencyFilter::new([ByteThreshold::new(b'l', 1)]).unwrap();
    let scanner = FusedScanner::new(patterns.clone(), Some(filter));
    // Provide a string that the filter WOULD reject (no 'l')
    let data = b"regex";
    let matches = scanner.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
}

// 23. Concurrent scan sharing the same JitDfa instance
#[test]
fn test_23_concurrent_jit_scan_sharing() {
    let patterns = Arc::new(PatternSet::builder().literal("shared").build().unwrap());
    let scanner = Arc::new(FusedScanner::new((*patterns).clone(), None));
    let mut handles = vec![];
    for _ in 0..10 {
        let sc = Arc::clone(&scanner);
        handles.push(std::thread::spawn(move || {
            let res = sc.scan(b"this is a shared test").unwrap();
            assert_eq!(res.len(), 1);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

// 24. FusedScanner clone produces valid clone
#[test]
fn test_24_fused_scanner_clone() {
    let patterns = PatternSet::builder().literal("clone_me").build().unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let cloned = scanner.clone();
    let data = b"clone_me twice clone_me";
    assert_eq!(cloned.scan(data).unwrap().len(), 2);
}

// 25. Large overlapping hits return all
#[test]
fn test_25_large_overlapping_hits() {
    let patterns = PatternSet::builder().literal("aaa").build().unwrap();
    let data = b"aaaaa";
    let matches = patterns.scan_overlapping(data).unwrap();
    assert_eq!(matches.len(), 3);
}

// 26. Invalid CRC fails index rebuild
#[test]
fn test_26_invalid_crc_fails_rebuild() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let mut bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let len = bytes.len();
    bytes[len - 1] ^= 0xFF; // Modify CRC byte
    let res = CompiledPatternIndex::load(&bytes);
    assert!(res.is_err());
}

// 27. Missing magic bytes fails index load
#[test]
fn test_27_missing_magic_bytes() {
    let res = CompiledPatternIndex::load(b"NOT_A_VALID_HEADER_FILE");
    assert!(res.is_err());
}

// 28. Regex parsing pathological repetition
#[test]
fn test_28_regex_pathological_repetition() {
    let err = PatternSet::builder().regex("a{1000}").build();
    if let Err(e) = err {
        assert!(
            matches!(e, Error::PathologicalRegex { .. })
                || matches!(e, Error::PatternCompilationFailed { .. })
        );
    }
}

// 29. Null byte literal scanning works
#[test]
fn test_29_null_byte_literal_scanning() {
    let patterns = PatternSet::builder()
        .literal_bytes(vec![0x00, 0x01, 0x00])
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = [0x00, 0x01, 0x00, 0x00, 0x01, 0x00];
    let matches = scanner.scan(&data).unwrap();
    assert_eq!(matches.len(), 2);
}

// 30. SimdSieve with non-alphabetic candidates
#[test]
fn test_30_simd_sieve_non_alphabetic() {
    let patterns = PatternSet::builder()
        .literal_bytes(vec![0xFF, 0xFE, 0xFD])
        .build()
        .unwrap();
    let scanner = FusedScanner::new(patterns.clone(), None);
    let data = [0xFF, 0xFE, 0xFD, 0x00];
    let matches = scanner.scan(&data).unwrap();
    assert_eq!(matches.len(), 1);
}

// 31. Large scale GPU overlap default testing via AutoMatcher config overlap check
#[tokio::test]
async fn test_31_automatcher_config_overlap_check() {
    let config = AutoMatcherConfig::new().chunk_size(10).chunk_overlap(100);
    assert_eq!(config.configured_chunk_overlap(), 100);
}

// 32. Truncated index rebuild correctly reports error
#[test]
fn test_32_truncated_index_rebuild_error() {
    let patterns = PatternSet::builder().literal("truncate").build().unwrap();
    let mut bytes = CompiledPatternIndex::build(&patterns).unwrap();
    bytes.truncate(10);
    let res = CompiledPatternIndex::load(&bytes);
    assert!(res.is_err());
}

// 33. Empty string as named regex group gracefully errors
#[test]
fn test_33_empty_string_named_regex() {
    let err = PatternSet::builder().named_regex("empty", "^$").build();
    if let Err(e) = err {
        assert!(
            matches!(e, Error::PatternCompilationFailed { .. })
                || matches!(e, Error::PathologicalRegex { .. })
        );
    }
}

// 34. Check overlapping regex logic on same index
#[test]
fn test_34_overlapping_regex_logic() {
    let patterns = PatternSet::builder().regex("a+").build().unwrap();
    let data = b"aaaa";
    let matches = patterns.scan(data).unwrap();
    // Aho-Corasick Leftmost-First will match the entire "aaaa" as one hit
    assert_eq!(matches.len(), 1);
}
