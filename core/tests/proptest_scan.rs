#![cfg(feature = "gpu")]
//! Property-based tests for warpstate using proptest.
//!
//! These tests verify fundamental invariants of the scanning engine:
//! - Match positions are valid
//! - Reported matches contain actual pattern bytes
//! - Match ordering guarantees
//! - Non-overlapping vs overlapping semantics
//! - CPU/GPU parity
//! - Streaming equivalence
//! - Batch processing correctness

#![cfg(not(miri))]

use proptest::collection::vec;
use proptest::prelude::*;
use warpstate::{batch, Match, PatternSet, StreamScanner};

// =============================================================================
// Property 1: Match positions are valid
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// For any input bytes and any literal pattern: scan() result positions
    /// are all valid (start < end, end <= input.len()).
    #[test]
    fn prop_match_positions_valid(
        input in vec(any::<u8>(), 0..10_000),
        pattern in vec(any::<u8>(), 1..64)
    ) {
        let patterns = match PatternSet::builder().literal_bytes(pattern.clone()).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = patterns.scan(&input).unwrap();
        let input_len = input.len() as u32;

        for m in &matches {
            prop_assert!(m.start < m.end,
                "Match start {} must be < end {}", m.start, m.end);
            prop_assert!(m.end <= input_len,
                "Match end {} must be <= input len {}", m.end, input_len);
        }
    }
}

// =============================================================================
// Property 2: Matches contain actual pattern bytes
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// For any input and pattern: every reported match actually contains
    /// the pattern bytes at [start..end].
    #[test]
    fn prop_matches_contain_pattern_bytes(
        input in vec(any::<u8>(), 0..10_000),
        pattern in vec(any::<u8>(), 1..64)
    ) {
        let patterns = match PatternSet::builder().literal_bytes(pattern.clone()).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = patterns.scan(&input).unwrap();

        for m in &matches {
            let matched_bytes = &input[m.start as usize..m.end as usize];
            prop_assert_eq!(matched_bytes, pattern.as_slice(),
                "Match at [{}..{}) does not contain pattern bytes", m.start, m.end);
        }
    }
}

// =============================================================================
// Property 3: Matches are sorted by start position
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// For any input and pattern: scan() matches are sorted by start position.
    #[test]
    fn prop_matches_sorted_by_start(
        input in vec(any::<u8>(), 0..10_000),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..10)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = patterns.scan(&input).unwrap();

        for window in matches.windows(2) {
            prop_assert!(
                window[0].start <= window[1].start,
                "Matches not sorted: match at {} comes before match at {}",
                window[0].start, window[1].start
            );
        }
    }
}

// =============================================================================
// Property 4: Non-overlapping mode produces non-overlapping matches
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// For any input and pattern: scan() matches don't overlap (non-overlapping mode).
    #[test]
    fn prop_non_overlapping_matches_no_overlap(
        input in vec(any::<u8>(), 0..10_000),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..10)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = patterns.scan(&input).unwrap();

        for window in matches.windows(2) {
            // In non-overlapping mode, the next match must start at or after
            // the previous match's end
            prop_assert!(
                window[1].start >= window[0].end,
                "Overlapping matches detected: [{}, {}) overlaps with [{}, {})",
                window[0].start, window[0].end, window[1].start, window[1].end
            );
        }
    }
}

// =============================================================================
// Property 5: Overlapping mode produces >= matches than non-overlapping
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// For any input and pattern: scan_overlapping() >= scan() in match count.
    #[test]
    fn prop_overlapping_gte_non_overlapping(
        input in vec(any::<u8>(), 0..10_000),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..10)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let non_overlapping = patterns.scan(&input).unwrap();
        let overlapping = patterns.scan_overlapping(&input).unwrap();

        prop_assert!(
            overlapping.len() >= non_overlapping.len(),
            "Overlapping scan produced {} matches but non-overlapping produced {}",
            overlapping.len(), non_overlapping.len()
        );

        // Also verify: every non-overlapping match should exist in overlapping
        for no_match in &non_overlapping {
            let found = overlapping.iter().any(|o| {
                o.pattern_id == no_match.pattern_id
                    && o.start == no_match.start
                    && o.end == no_match.end
            });
            prop_assert!(
                found,
                "Non-overlapping match {:?} not found in overlapping results",
                no_match
            );
        }
    }
}

// =============================================================================
// Property 6: CPU scan == GPU scan (when GPU available)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// CPU scan == GPU scan for random (input, pattern) pairs (when GPU available).
    #[test]
    fn prop_cpu_gpu_parity(
        input in vec(any::<u8>(), 0..5000),
        patterns_data in vec(vec(any::<u8>(), 1..16), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let cpu_matches = patterns.scan(&input).unwrap();

        // Try GPU scan if available
        let gpu_matches = pollster::block_on(async {
            use warpstate::GpuMatcher;
            match GpuMatcher::new(&patterns).await {
                Ok(gpu) => gpu.scan(&input).await.ok(),
                Err(_) => None,
            }
        });

        if let Some(gpu) = gpu_matches {
            prop_assert_eq!(
                cpu_matches.len(), gpu.len(),
                "CPU and GPU match counts differ"
            );

            for (i, (cpu, gpu)) in cpu_matches.iter().zip(gpu.iter()).enumerate() {
                prop_assert_eq!(
                    cpu.pattern_id, gpu.pattern_id,
                    "Pattern ID mismatch at index {}", i
                );
                prop_assert_eq!(
                    cpu.start, gpu.start,
                    "Start offset mismatch at index {}", i
                );
                prop_assert_eq!(
                    cpu.end, gpu.end,
                    "End offset mismatch at index {}", i
                );
            }
        }
    }
}

// =============================================================================
// Property 7: StreamScanner produces same total matches as scan()
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// StreamScanner.feed() produces same total matches as scan() on concatenated input.
    #[test]
    fn prop_stream_scanner_equivalence(
        chunks in vec(vec(any::<u8>(), 0..1000), 1..20),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        // Concatenate all chunks
        let concatenated: Vec<u8> = chunks.iter().flatten().copied().collect();

        // Scan concatenated data directly
        let direct_matches = patterns.scan(&concatenated).unwrap();

        // Scan via StreamScanner
        let mut stream_scanner = StreamScanner::new(&patterns).unwrap();
        let mut stream_matches: Vec<Match> = Vec::new();
        for chunk in &chunks {
            stream_matches.extend(stream_scanner.feed(chunk).unwrap());
        }
        stream_matches.extend(stream_scanner.finish().unwrap());

        // Sort both by (start, pattern_id, end) for comparison
        let sort_key = |m: &Match| (m.start, m.pattern_id, m.end);
        let mut direct_sorted = direct_matches;
        let mut stream_sorted = stream_matches;
        direct_sorted.sort_unstable_by_key(sort_key);
        stream_sorted.sort_unstable_by_key(sort_key);

        prop_assert_eq!(
            direct_sorted.len(), stream_sorted.len(),
            "Direct scan produced {} matches but stream scanner produced {}",
            direct_sorted.len(), stream_sorted.len()
        );

        for (i, (direct, stream)) in direct_sorted.iter().zip(stream_sorted.iter()).enumerate() {
            prop_assert_eq!(
                direct.pattern_id, stream.pattern_id,
                "Pattern ID mismatch at index {}", i
            );
            prop_assert_eq!(
                direct.start, stream.start,
                "Start offset mismatch at index {}", i
            );
            prop_assert_eq!(
                direct.end, stream.end,
                "End offset mismatch at index {}", i
            );
        }
    }
}

// =============================================================================
// Property 8: Batch coalesce→scan→decoalesce matches individual scanning
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Batch coalesce→scan→decoalesce produces same matches as scanning items individually.
    #[test]
    fn prop_batch_roundtrip_equivalence(
        items_data in vec(
            (any::<u64>(), vec(any::<u8>(), 0..500)),
            1..20
        ),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        // Build batch items
        let items: Vec<batch::ScanItem> = items_data
            .iter()
            .map(|(id, data)| batch::ScanItem { id: *id, data: data.as_slice() })
            .collect();

        // Scan individually
        let individual_matches = batch::scan_batch_cpu(&patterns, items.clone()).unwrap();

        // Coalesce → scan → decoalesce
        let map = batch::coalesce(&items).unwrap();
        let global_matches = patterns.scan(&map.buffer).unwrap();
        let decoalesced_matches = batch::decoalesce(&map, global_matches);

        prop_assert_eq!(
            individual_matches.len(), decoalesced_matches.len(),
            "Individual scan produced {} matches but batch produced {}",
            individual_matches.len(), decoalesced_matches.len()
        );

        for (i, (ind, dec)) in individual_matches.iter().zip(decoalesced_matches.iter()).enumerate() {
            prop_assert_eq!(
                ind.source_id, dec.source_id,
                "Source ID mismatch at index {}", i
            );
            prop_assert_eq!(
                ind.matched.pattern_id, dec.matched.pattern_id,
                "Pattern ID mismatch at index {}", i
            );
            prop_assert_eq!(
                ind.matched.start, dec.matched.start,
                "Start offset mismatch at index {}", i
            );
            prop_assert_eq!(
                ind.matched.end, dec.matched.end,
                "End offset mismatch at index {}", i
            );
        }
    }
}

// =============================================================================
// Property 9: Regex scan doesn't panic on random binary input
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// For any regex that compiles: scan doesn't panic (even on random binary input).
    #[test]
    fn prop_regex_no_panic(
        input in vec(any::<u8>(), 0..5000),
        regex_patterns in vec("[a-zA-Z0-9.*+?^${}()|\\[\\]\\\\]*", 1..5)
    ) {
        // Build regex patterns - skip if compilation fails
        let mut builder = PatternSet::builder();
        for pattern in &regex_patterns {
            // Filter out empty patterns and pathological ones
            if pattern.is_empty() || pattern.contains("++") || pattern.contains("**") {
                return Ok(());
            }
            builder = builder.regex(pattern);
        }

        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()), // Compilation failed, skip this case
        };

        // This should not panic, regardless of input
        let _ = patterns.scan(&input);
        let _ = patterns.scan_overlapping(&input);
    }
}

// =============================================================================
// Property 10: Empty pattern set produces empty results, no panic
// =============================================================================

#[test]
fn prop_empty_pattern_set_returns_error() {
    let result = PatternSet::builder().build();
    assert!(
        result.is_err(),
        "Empty pattern set should return error, not empty results"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// PatternSet::builder() with 0 patterns → empty results, no panic.
    #[test]
    fn prop_empty_builder_no_panic(input in vec(any::<u8>(), 0..1000)) {
        // Attempting to build with no patterns should fail gracefully
        let result = PatternSet::builder().build();
        prop_assert!(result.is_err(), "Empty pattern set should return error");

        // Also verify that scanning with single pattern that doesn't match
        // returns empty results (not an error)
        let patterns = PatternSet::builder()
            .literal_bytes(vec![0xFF, 0xFE, 0xFD])
            .build()
            .unwrap();

        let matches = patterns.scan(&input).unwrap();
        // Should either have matches or be empty, but never error
        // (unless input is too large which is handled separately)
        prop_assert!(matches.iter().all(|m| m.start < m.end));
    }
}

// =============================================================================
// Additional edge case properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Case-insensitive scanning finds matches regardless of case.
    #[test]
    fn prop_case_insensitive_finds_variations(
        input in "[a-zA-Z]{0,1000}",
        pattern in "[a-z]{1,20}"
    ) {
        let patterns_ci = PatternSet::builder()
            .literal(&pattern)
            .case_insensitive(true)
            .build();

        let patterns_cs = PatternSet::builder()
            .literal(&pattern)
            .case_insensitive(false)
            .build();

        let patterns_ci = match patterns_ci {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let patterns_cs = match patterns_cs {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let input_bytes = input.as_bytes();
        let ci_matches = patterns_ci.scan(input_bytes).unwrap();
        let cs_matches = patterns_cs.scan(input_bytes).unwrap();

        // Case-insensitive should find at least as many matches as case-sensitive
        prop_assert!(
            ci_matches.len() >= cs_matches.len(),
            "Case-insensitive found {} matches but case-sensitive found {}",
            ci_matches.len(), cs_matches.len()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// scan_with produces the same results as scan.
    #[test]
    fn prop_scan_with_equivalence(
        input in vec(any::<u8>(), 0..5000),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..10)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let scan_matches = patterns.scan(&input).unwrap();

        let mut with_matches: Vec<Match> = Vec::new();
        patterns.scan_with(&input, |m| {
            with_matches.push(m);
            true
        }).unwrap();

        prop_assert_eq!(
            scan_matches.len(), with_matches.len(),
            "scan and scan_with produced different match counts"
        );

        // Sort both by (start, pattern_id, end) for comparison
        let sort_key = |m: &Match| (m.start, m.pattern_id, m.end);
        let mut scan_sorted = scan_matches;
        let mut with_sorted = with_matches;
        scan_sorted.sort_unstable_by_key(sort_key);
        with_sorted.sort_unstable_by_key(sort_key);

        for (i, (s, w)) in scan_sorted.iter().zip(with_sorted.iter()).enumerate() {
            prop_assert_eq!(s.pattern_id, w.pattern_id, "Pattern ID mismatch at {}", i);
            prop_assert_eq!(s.start, w.start, "Start mismatch at {}", i);
            prop_assert_eq!(s.end, w.end, "End mismatch at {}", i);
        }
    }

    /// scan_count produces the same count as scan().len().
    #[test]
    fn prop_scan_count_equivalence(
        input in vec(any::<u8>(), 0..5000),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..10)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let scan_matches = patterns.scan(&input).unwrap();
        let scan_count = patterns.scan_count(&input).unwrap();

        prop_assert_eq!(
            scan_count,
            scan_matches.len(),
            "scan_count and scan().len() produced different counts"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Empty patterns should be rejected during build.
    #[test]
    fn prop_empty_pattern_rejected(
        prefix in vec(any::<u8>(), 0..10),
        suffix in vec(any::<u8>(), 0..10)
    ) {
        // Try to build with an empty literal in the middle
        let result = PatternSet::builder()
            .literal_bytes(prefix.clone())
            .literal("")
            .literal_bytes(suffix.clone())
            .build();

        prop_assert!(
            result.is_err(),
            "Pattern set with empty pattern should fail to build"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Pattern matching is idempotent: scanning the same input twice
    /// produces identical results.
    #[test]
    fn prop_scan_idempotent(
        input in vec(any::<u8>(), 0..5000),
        patterns_data in vec(vec(any::<u8>(), 1..32), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let patterns = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches1 = patterns.scan(&input).unwrap();
        let matches2 = patterns.scan(&input).unwrap();

        prop_assert_eq!(
            matches1.len(), matches2.len(),
            "Two scans produced different match counts"
        );

        for (i, (m1, m2)) in matches1.iter().zip(matches2.iter()).enumerate() {
            prop_assert_eq!(m1.pattern_id, m2.pattern_id, "Pattern ID mismatch at {}", i);
            prop_assert_eq!(m1.start, m2.start, "Start mismatch at {}", i);
            prop_assert_eq!(m1.end, m2.end, "End mismatch at {}", i);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Substring property: if a pattern matches at a position in the full input,
    /// it should also match in a window containing that match.
    #[test]
    fn prop_match_window_consistency(
        prefix in vec(any::<u8>(), 0..100),
        pattern in vec(any::<u8>(), 1..32),
        suffix in vec(any::<u8>(), 0..100)
    ) {
        let patterns = match PatternSet::builder().literal_bytes(pattern.clone()).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        // Build input with pattern in the middle
        let mut input = prefix.clone();
        input.extend_from_slice(&pattern);
        input.extend_from_slice(&suffix);

        let full_matches = patterns.scan(&input).unwrap();

        // Build window containing just the pattern
        let window_start = prefix.len();
        let window_end = prefix.len() + pattern.len();
        let window = &input[window_start..window_end];
        let window_matches = patterns.scan(window).unwrap();

        // Pattern should match in the window exactly once
        prop_assert_eq!(
            window_matches.len(), 1,
            "Pattern should match exactly once in its own window"
        );

        // The full input should have at least one match
        prop_assert!(
            !full_matches.is_empty(),
            "Pattern should match at least once in full input containing it"
        );

        // Verify the match positions are consistent
        if let Some(first_match) = full_matches.first() {
            prop_assert_eq!(
                first_match.end - first_match.start,
                pattern.len() as u32,
                "Match length should equal pattern length"
            );
        }
    }
}
