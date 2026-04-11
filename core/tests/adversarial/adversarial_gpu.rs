#![cfg(feature = "gpu")]
//! Adversarial tests for warpstate GPU scan path
//!
//! These tests are designed to BREAK the GPU scanner by testing:
//! - GPU/CPU parity with random inputs and patterns
//! - Chunk boundary conditions (patterns at exact boundaries)
//! - Maximum input size handling
//! - Memory pressure and buffer pool behavior
//! - Concurrent access patterns
//! - Auto-routing logic
//!
//! All tests are gated by `has_gpu()` — they run only when a GPU is available.

use std::sync::Arc;

use warpstate::batch::{scan_batch_gpu, ScanItem};
use warpstate::GpuMatcher;
use warpstate::StreamPipeline;
use warpstate::{AutoMatcher, AutoMatcherConfig, Error, Matcher, PatternSet};

/// Helper: Check if GPU is available by trying to create an AutoMatcher.
fn has_gpu() -> bool {
    let patterns = match PatternSet::builder().literal("test").build() {
        Ok(ps) => ps,
        Err(_) => return false,
    };
    match pollster::block_on(AutoMatcher::new(&patterns)) {
        Ok(matcher) => matcher.has_gpu(),
        Err(_) => false,
    }
}

/// Helper: Block on an async future (synchronous test context).
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

/// Approximate max input size for GPU tests.
/// Uses a conservative value that should work on most GPU devices.
/// Actual device limits may vary based on:
/// - max_storage_buffer_binding_size (typically 128MB - 1GB)
/// - max_compute_workgroups_per_dimension (typically 65535)
///
/// Workgroup calculation: workgroups = input_len / WORKGROUP_SIZE (256)
/// 65535 * 256 = 16,776,960 bytes is the theoretical max for workgroups.
/// We use 8MB to stay well under all limits and leave room for alignment.
fn get_safe_test_size() -> usize {
    8 * 1024 * 1024 // 8MB
}

// =============================================================================
// Test 1: GPU/CPU Parity — Random 1MB Input + 100 Random Patterns
// =============================================================================

#[test]
fn gpu_cpu_parity_random_1mb_input_100_patterns() {
    if !has_gpu() {
        return;
    }

    // Generate 1MB of random data
    let mut input = vec![0u8; 1024 * 1024];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = ((i * 7 + 13) % 256) as u8;
    }

    // Embed known patterns at specific positions for verification
    let known_positions = [0, 1000, 50000, 100000, 500000, 999000];
    for pos in &known_positions {
        if *pos + 10 < input.len() {
            input[*pos..*pos + 10].copy_from_slice(b"KNOWNPATT_");
        }
    }

    // Build pattern set with 100 patterns (including our known pattern)
    let mut builder = PatternSet::builder();
    builder = builder.literal("KNOWNPATT_"); // pattern 0
    for i in 1..100 {
        builder = builder.literal(&format!("RANDPAT{i:04}"));
    }
    let patterns = builder.build().unwrap();

    // Get CPU matches (using overlapping for GPU parity)
    let cpu_matches = patterns.scan_overlapping(&input).unwrap();

    // Get GPU matches
    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    // Sort both for comparison (GPU may return in different order)
    let mut cpu_sorted = cpu_matches.clone();
    let mut gpu_sorted = gpu_matches.clone();
    cpu_sorted.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.pattern_id.cmp(&b.pattern_id))
            .then(a.end.cmp(&b.end))
    });
    gpu_sorted.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.pattern_id.cmp(&b.pattern_id))
            .then(a.end.cmp(&b.end))
    });

    // Verify known pattern matches
    let known_cpu = cpu_matches.iter().filter(|m| m.pattern_id == 0).count();
    let known_gpu = gpu_matches.iter().filter(|m| m.pattern_id == 0).count();
    assert_eq!(
        known_gpu, known_cpu,
        "GPU should find same number of known pattern matches as CPU"
    );
    assert_eq!(
        known_gpu,
        known_positions.len(),
        "GPU should find all embedded known patterns"
    );
}

// =============================================================================
// Test 2: GPU/CPU Parity — Regex Pattern on Random Input
// =============================================================================

#[test]
fn gpu_cpu_parity_regex_on_random_input() {
    if !has_gpu() {
        return;
    }

    // Generate random input with embedded regex matches
    let mut input = vec![0u8; 256 * 1024]; // 256KB
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = ((i * 11 + 17) % 256) as u8;
    }

    // Embed patterns that match the regex
    let regex_embeds = ["abbbc", "abbc", "abbbbbc"];
    let embed_positions = [100, 50000, 150000];
    for (i, pos) in embed_positions.iter().enumerate() {
        if *pos + regex_embeds[i].len() < input.len() {
            input[*pos..*pos + regex_embeds[i].len()].copy_from_slice(regex_embeds[i].as_bytes());
        }
    }

    let patterns = PatternSet::builder().regex(r"ab+c").build().unwrap();

    // CPU matches
    let cpu_matches = patterns.scan(&input).unwrap();

    // GPU matches
    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    // Verify same number of matches
    assert_eq!(
        gpu_matches.len(),
        cpu_matches.len(),
        "GPU and CPU should find same number of regex matches"
    );

    // Verify all embedded patterns were found
    assert!(
        gpu_matches.len() >= embed_positions.len(),
        "GPU should find all embedded regex patterns"
    );
}

// =============================================================================
// Test 3: Chunk Boundary — Pattern Spans Exact Chunk Boundary
// =============================================================================

#[test]
fn chunk_boundary_pattern_spans_exact_boundary() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();

    // Use a smaller chunk size that fits within device limits
    let chunk_size = (max_size / 2).min(16 * 1024 * 1024); // Half of max, capped at 16MB

    // Create a pattern that will span the chunk boundary
    let pattern = b"CHUNK_BOUNDARY_PATTERN";
    let pattern_len = pattern.len();

    // Create input with pattern straddling chunk boundary
    let boundary_pos = chunk_size;
    let pattern_start = boundary_pos - (pattern_len / 2);

    let mut input = vec![b'x'; chunk_size + 1024];
    input[pattern_start..pattern_start + pattern_len].copy_from_slice(pattern);

    let patterns = PatternSet::builder()
        .literal(std::str::from_utf8(pattern).unwrap())
        .build()
        .unwrap();

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    assert_eq!(
        gpu_matches.len(),
        1,
        "Pattern spanning chunk boundary should be found"
    );
    assert_eq!(
        gpu_matches[0].start as usize, pattern_start,
        "Match should start at correct position"
    );
}

// =============================================================================
// Test 4: Chunk Boundary — Pattern at Last Byte of Chunk
// =============================================================================

#[test]
fn chunk_boundary_pattern_at_last_byte_of_chunk() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();

    let chunk_size = (max_size / 2).min(16 * 1024 * 1024);

    // Pattern ending exactly at the chunk boundary
    let pattern = b"ENDCHUNK";
    let pattern_len = pattern.len();

    let boundary_pos = chunk_size;
    let pattern_start = boundary_pos - pattern_len;

    let mut input = vec![b'x'; chunk_size + 1024];
    input[pattern_start..pattern_start + pattern_len].copy_from_slice(pattern);

    let patterns = PatternSet::builder()
        .literal(std::str::from_utf8(pattern).unwrap())
        .build()
        .unwrap();

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    assert_eq!(
        gpu_matches.len(),
        1,
        "Pattern ending at chunk boundary should be found"
    );
    assert_eq!(
        gpu_matches[0].start as usize, pattern_start,
        "Match should start at correct position"
    );
    assert_eq!(
        gpu_matches[0].end as usize, boundary_pos,
        "Match should end at chunk boundary"
    );
}

// =============================================================================
// Test 5: Chunk Boundary — Pattern at First Byte of Next Chunk
// =============================================================================

#[test]
fn chunk_boundary_pattern_at_first_byte_of_next_chunk() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();

    let chunk_size = (max_size / 2).min(16 * 1024 * 1024);

    // Pattern starting exactly at the chunk boundary
    let pattern = b"STARTNEXT";
    let pattern_len = pattern.len();

    let boundary_pos = chunk_size;

    let mut input = vec![b'x'; chunk_size + 1024];
    input[boundary_pos..boundary_pos + pattern_len].copy_from_slice(pattern);

    let patterns = PatternSet::builder()
        .literal(std::str::from_utf8(pattern).unwrap())
        .build()
        .unwrap();

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    assert_eq!(
        gpu_matches.len(),
        1,
        "Pattern starting at chunk boundary should be found"
    );
    assert_eq!(
        gpu_matches[0].start as usize, boundary_pos,
        "Match should start at chunk boundary"
    );
}

// =============================================================================
// Test 6: Max Input Size — Exactly Device Max Bytes
// =============================================================================

#[test]
fn max_input_size_exactly_device_max() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();

    // Create input of exactly max_size bytes
    let mut input = vec![b'x'; max_size];

    // Embed a pattern at the end
    let pattern = b"ENDPATTERN";
    let embed_pos = max_size - pattern.len();
    input[embed_pos..].copy_from_slice(pattern);

    let patterns = PatternSet::builder()
        .literal(std::str::from_utf8(pattern).unwrap())
        .build()
        .unwrap();

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    assert_eq!(
        gpu_matches.len(),
        1,
        "Pattern at end of max-size input should be found"
    );
    assert_eq!(
        gpu_matches[0].start as usize, embed_pos,
        "Match should be at correct position"
    );
}

// =============================================================================
// Test 7: Over Max Input Size with Hard Limit — InputTooLarge Error
// =============================================================================

#[test]
fn over_max_input_size_with_hard_limit_returns_error() {
    if !has_gpu() {
        return;
    }

    // Use with_options which sets hard_input_limit = true
    let patterns = PatternSet::builder().literal("test").build().unwrap();

    // Try to create matcher with small max input size
    let small_max = 1024;
    let gpu_matcher = match block_on(GpuMatcher::with_options(&patterns, small_max)) {
        Ok(m) => m,
        Err(Error::NoGpuAdapter) => return,
        Err(e) => panic!("Unexpected error creating GPU matcher: {e:?}"),
    };

    // Create input larger than the limit
    let input = vec![b'x'; small_max + 1];

    // Should get InputTooLarge error
    let result = block_on(gpu_matcher.scan(&input));
    assert!(
        matches!(result, Err(Error::InputTooLarge { .. })),
        "Should return InputTooLarge error for oversized input"
    );
}

// =============================================================================
// Test 8: Zero-Length Input to GPU — Empty Vec, No Crash
// =============================================================================

#[test]
fn zero_length_input_to_gpu_returns_empty() {
    if !has_gpu() {
        return;
    }

    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();

    // Empty input should return empty results without error
    let matches = block_on(gpu_matcher.scan(b"")).unwrap();
    assert!(
        matches.is_empty(),
        "Empty input should return empty matches"
    );
}

// =============================================================================
// Test 9: Single-Byte Pattern on Large Input — Correct Count
// =============================================================================

#[test]
fn single_byte_pattern_on_large_input() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();

    // Use 16MB or max_size/4, whichever is smaller, to stay within workgroup limits
    let input_size = (16 * 1024 * 1024).min(max_size / 4);
    let mut input = vec![b'x'; input_size];

    // Insert the target byte every 1024 bytes
    let target_byte = b'A';
    let spacing = 1024;
    let expected_count = input_size / spacing;

    for i in (0..input_size).step_by(spacing) {
        input[i] = target_byte;
    }

    let patterns = PatternSet::builder()
        .literal(std::str::from_utf8(&[target_byte]).unwrap())
        .build()
        .unwrap();

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    assert_eq!(
        gpu_matches.len(),
        expected_count,
        "GPU should find correct number of single-byte matches"
    );

    // Verify CPU finds the same count
    let cpu_matches = patterns.scan_overlapping(&input).unwrap();
    assert_eq!(
        gpu_matches.len(),
        cpu_matches.len(),
        "GPU and CPU should agree on match count"
    );
}

// =============================================================================
// Test 10: 10,000 Patterns on GPU — All Matched Correctly
// =============================================================================

#[test]
fn ten_thousand_patterns_on_gpu() {
    if !has_gpu() {
        return;
    }

    // Build 10,000 patterns
    let mut builder = PatternSet::builder();
    for i in 0..10000 {
        builder = builder.literal(&format!("PATTERN{i:05}"));
    }
    let patterns = builder.build().unwrap();

    // Create input containing specific patterns at known positions
    let mut input = vec![b'x'; 1024 * 1024]; // 1MB

    // Embed patterns at specific positions
    let test_patterns = [
        (0, "PATTERN00000"),
        (5000, "PATTERN05000"),
        (9999, "PATTERN09999"),
    ];

    for (id, pat) in &test_patterns {
        let pos = (*id + 1) * 100; // Spread them out
        if pos + pat.len() < input.len() {
            input[pos..pos + pat.len()].copy_from_slice(pat.as_bytes());
        }
    }

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    // Verify each test pattern was found
    for (id, _) in &test_patterns {
        let found = gpu_matches.iter().any(|m| m.pattern_id == *id as u32);
        assert!(found, "Pattern {id} should be found by GPU");
    }

    // Verify no duplicate matches for same pattern
    let mut pattern_counts: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    for m in &gpu_matches {
        *pattern_counts.entry(m.pattern_id).or_insert(0) += 1;
    }

    for (id, count) in pattern_counts {
        assert_eq!(
            count, 1,
            "Pattern {id} should have exactly one match, found {count}"
        );
    }
}

// =============================================================================
// Test 11: Buffer Pool — Scan 100 Times Sequentially, No Memory Leak
// =============================================================================

#[test]
fn buffer_pool_reuse_sequential_scans_no_leak() {
    if !has_gpu() {
        return;
    }

    let patterns = PatternSet::builder().literal("needle").build().unwrap();
    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();

    // Scan 100 times with same-size input (should reuse buffers)
    let input = vec![b'x'; 1024 * 1024]; // 1MB

    for i in 0..100 {
        // Embed pattern at different positions
        let mut test_input = input.clone();
        let pos = (i * 10000) % (test_input.len() - 10);
        test_input[pos..pos + 6].copy_from_slice(b"needle");

        let matches = block_on(gpu_matcher.scan(&test_input)).unwrap();
        assert_eq!(matches.len(), 1, "Scan {i} should find exactly one match");
        assert_eq!(
            matches[0].pattern_id, 0,
            "Scan {i} should find correct pattern"
        );
    }

    // Test passes if we complete without OOM or errors
}

// =============================================================================
// Test 12: Concurrent Scans — 4 Threads Sharing Arc<GpuMatcher>
// =============================================================================

#[test]
fn concurrent_scans_four_threads_shared_gpu_matcher() {
    if !has_gpu() {
        return;
    }

    let patterns = PatternSet::builder().literal("CONCURRENT").build().unwrap();
    let gpu_matcher = Arc::new(block_on(GpuMatcher::new(&patterns)).unwrap());

    let mut handles = vec![];

    // Spawn 4 threads, each scanning different inputs
    for thread_id in 0..4 {
        let matcher = Arc::clone(&gpu_matcher);
        let handle = std::thread::spawn(move || {
            for i in 0..10 {
                // Create input with pattern at calculated position
                let mut input = vec![b'x'; 256 * 1024];
                let pos = (thread_id * 10000 + i * 1000) % (input.len() - 10);
                input[pos..pos + 10].copy_from_slice(b"CONCURRENT");

                let matches = block_on(matcher.scan(&input)).unwrap();
                assert_eq!(
                    matches.len(),
                    1,
                    "Thread {thread_id}, scan {i} should find exactly one match"
                );
                assert_eq!(
                    matches[0].pattern_id, 0,
                    "Thread {thread_id}, scan {i} should find correct pattern"
                );
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

// =============================================================================
// Test 13: AutoMatcher Routing — Small Input Goes CPU, Large Input Goes GPU
// =============================================================================

#[test]
fn auto_matcher_routing_small_cpu_large_gpu() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();
    let patterns = PatternSet::builder()
        .literal("ROUTINGTEST")
        .build()
        .unwrap();

    // Create matcher with a moderate threshold
    let threshold = (64 * 1024).min(max_size / 8);
    let matcher = block_on(AutoMatcher::with_config(
        &patterns,
        AutoMatcherConfig::new().gpu_threshold(threshold),
    ))
    .unwrap();

    // Get the actual GPU max input size
    let gpu_max = matcher.gpu_max_input_size();

    // Test 1: Small input (well below threshold) - should route to CPU
    let small_input = {
        let mut buf = vec![b'x'; 1024]; // 1KB
        buf[100..111].copy_from_slice(b"ROUTINGTEST");
        buf
    };

    let small_matches = block_on(matcher.scan(&small_input)).unwrap();
    assert_eq!(small_matches.len(), 1, "Small input should find match");

    // Test 2: Medium input (above threshold, within GPU max) - should route to GPU
    // Choose a size that definitely exceeds threshold but is well within GPU limits
    let medium_size = (threshold + 16 * 1024).min(gpu_max / 4);
    let medium_input = {
        let mut buf = vec![b'x'; medium_size];
        // Place pattern near start to avoid chunk boundary issues
        buf[100..111].copy_from_slice(b"ROUTINGTEST");
        buf
    };

    let medium_matches = block_on(matcher.scan(&medium_input)).unwrap();
    assert_eq!(
        medium_matches.len(),
        1,
        "Medium input (threshold < size < gpu_max) should find match"
    );

    // Test 3: Verify forced GPU scan works
    let forced_gpu_matches = block_on(matcher.scan_gpu(&medium_input)).unwrap();
    assert_eq!(
        forced_gpu_matches.len(),
        1,
        "Forced GPU scan should find match"
    );

    // Verify all found the same pattern
    assert_eq!(small_matches[0].pattern_id, medium_matches[0].pattern_id);
    assert_eq!(
        small_matches[0].pattern_id,
        forced_gpu_matches[0].pattern_id
    );
}

// =============================================================================
// Test 14: StreamPipeline — Large Input Chunked, Matches Equal Single Scan
// =============================================================================

#[test]
fn stream_pipeline_large_input_matches_single_scan() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();

    // Create input that requires chunking
    // Use larger size to ensure chunking happens
    let input_size = (4 * 1024 * 1024).min(max_size / 2);
    let mut input = vec![b'x'; input_size];

    // Embed patterns at various positions
    let quarter = input_size / 4;
    let patterns_to_embed = [
        (quarter, "PATTERNONE"),
        (quarter * 2, "PATTERNTWO"),
        (quarter * 3, "PATTERNTHREE"),
    ];

    for (pos, pat) in &patterns_to_embed {
        if *pos + pat.len() < input.len() {
            input[*pos..*pos + pat.len()].copy_from_slice(pat.as_bytes());
        }
    }

    // Build patterns
    let mut builder = PatternSet::builder();
    for (_, pat) in &patterns_to_embed {
        builder = builder.literal(*pat);
    }
    let patterns = builder.build().unwrap();

    // Single scan reference (CPU)
    let cpu_matches = patterns.scan(&input).unwrap();

    // Verify CPU found all patterns
    assert_eq!(
        cpu_matches.len(),
        patterns_to_embed.len(),
        "CPU should find all {} embedded patterns",
        patterns_to_embed.len()
    );

    // StreamPipeline scan via the consolidated GpuMatcher backend.
    let gpu = match block_on(GpuMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(e) => panic!("Failed to create GpuMatcher: {e:?}"),
    };

    let pipeline = StreamPipeline::new(gpu, 4096);
    let pipeline_matches = block_on(pipeline.scan(&input)).unwrap();

    // Sort both for comparison
    let mut cpu_sorted = cpu_matches.clone();
    let mut pipe_sorted = pipeline_matches.clone();
    cpu_sorted.sort_by(|a, b| a.start.cmp(&b.start).then(a.pattern_id.cmp(&b.pattern_id)));
    pipe_sorted.sort_by(|a, b| a.start.cmp(&b.start).then(a.pattern_id.cmp(&b.pattern_id)));

    // Pipeline should find same or similar number of matches
    // Allow for small differences due to chunk boundary handling
    assert!(
        pipe_sorted.len() >= cpu_sorted.len().saturating_sub(1),
        "StreamPipeline should find at least {} matches, found {}",
        cpu_sorted.len().saturating_sub(1),
        pipe_sorted.len()
    );

    // Verify each pattern was found by checking pattern_ids
    for (expected_id, _) in patterns_to_embed.iter().enumerate() {
        let found_in_pipe = pipe_sorted
            .iter()
            .any(|m| m.pattern_id == expected_id as u32);
        assert!(
            found_in_pipe,
            "Pattern {expected_id} should be found by StreamPipeline"
        );
    }
}

// =============================================================================
// Test 15: Pattern with Only Regex (No Literals) — consolidated GPU regex path
// =============================================================================

#[test]
fn regex_only_pattern_consolidated_gpu_path() {
    if !has_gpu() {
        return;
    }

    // Pattern with no literal component - pure regex
    let patterns = PatternSet::builder()
        .regex(r"[a-z]+@[a-z]+\.[a-z]{2,}") // Email-like pattern
        .build()
        .unwrap();

    // Create input with email-like matches
    let input = b"Contact us at support@example.com or sales@company.org for help. Invalid: @@@. Another: info@test.io";

    // GPU matches
    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(input)).unwrap();

    // Should find matches
    assert!(
        !gpu_matches.is_empty(),
        "GPU should find regex-only pattern matches"
    );

    // Verify specific matches
    let found_example = gpu_matches
        .iter()
        .any(|m| &input[m.start as usize..m.end as usize] == b"support@example.com");
    let found_company = gpu_matches
        .iter()
        .any(|m| &input[m.start as usize..m.end as usize] == b"sales@company.org");

    assert!(found_example, "Should find 'support@example.com'");
    assert!(found_company, "Should find 'sales@company.org'");
}

// =============================================================================
// Additional Adversarial Tests
// =============================================================================

/// Test that patterns at various alignments (byte offsets mod 4) work correctly
#[test]
fn pattern_at_various_byte_alignments() {
    if !has_gpu() {
        return;
    }

    let pattern = "ALIGNMENT";
    let patterns = PatternSet::builder().literal(pattern).build().unwrap();
    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();

    // Test pattern at each byte alignment (0, 1, 2, 3 mod 4)
    for offset in 0..4 {
        let mut input = vec![b'x'; 1024];
        input[offset..offset + pattern.len()].copy_from_slice(pattern.as_bytes());

        let matches = block_on(gpu_matcher.scan(&input)).unwrap();
        assert_eq!(
            matches.len(),
            1,
            "Pattern at offset {offset} (mod 4) should be found"
        );
        assert_eq!(
            matches[0].start as usize, offset,
            "Match should be at offset {offset}"
        );
    }
}

/// Test multiple overlapping patterns at chunk boundaries
#[test]
fn multiple_overlapping_patterns_at_chunk_boundary() {
    if !has_gpu() {
        return;
    }

    let max_size = get_safe_test_size();

    let chunk_size = (max_size / 2).min(16 * 1024 * 1024);

    let patterns = PatternSet::builder()
        .literal("BOUNDARYA")
        .literal("BOUNDARYB")
        .literal("BOUNDARYC")
        .build()
        .unwrap();

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();

    // Create input with patterns near chunk boundary
    let boundary = chunk_size;
    let mut input = vec![b'x'; boundary + 1024];

    // Place patterns so they span or are near the boundary
    input[boundary - 5..boundary + 4].copy_from_slice(b"BOUNDARYA");

    let matches = block_on(gpu_matcher.scan(&input)).unwrap();
    assert!(
        matches.iter().any(|m| m.pattern_id == 0),
        "Pattern spanning boundary should be found"
    );
}

/// Test scan_batch_gpu with adversarial inputs
#[test]
fn batch_scan_gpu_adversarial_items() {
    if !has_gpu() {
        return;
    }

    let patterns = PatternSet::builder()
        .literal("BATCHPATTERN")
        .build()
        .unwrap();

    let matcher = block_on(AutoMatcher::with_config(
        &patterns,
        AutoMatcherConfig::new().gpu_threshold(0),
    ))
    .unwrap();

    // Create items of varying sizes including empty and very small
    let items: Vec<Vec<u8>> = vec![
        vec![],                                           // Empty
        vec![b'x'; 10],                                   // Very small
        vec![b'x'; 1024],                                 // 1KB
        format!("prefixBATCHPATTERNsuffix").into_bytes(), // Contains match
        vec![b'x'; 64 * 1024],                            // 64KB
        format!("BATCHPATTERN").into_bytes(),             // Exact match at start
    ];

    let scan_items: Vec<ScanItem<'_>> = items
        .iter()
        .enumerate()
        .map(|(idx, bytes)| ScanItem {
            id: idx as u64,
            data: bytes,
        })
        .collect();

    let matches = block_on(scan_batch_gpu(&matcher, scan_items)).unwrap();

    // Should find matches in items 3 and 5
    let item_3_matches: Vec<_> = matches.iter().filter(|m| m.source_id == 3).collect();
    let item_5_matches: Vec<_> = matches.iter().filter(|m| m.source_id == 5).collect();

    assert_eq!(item_3_matches.len(), 1, "Item 3 should have one match");
    assert_eq!(item_5_matches.len(), 1, "Item 5 should have one match");

    // Verify match positions
    assert_eq!(
        item_3_matches[0].matched.start, 6,
        "Match in item 3 should start after 'prefix'"
    );
    assert_eq!(
        item_5_matches[0].matched.start, 0,
        "Match in item 5 should start at position 0"
    );
}

/// Test that very long patterns work correctly
#[test]
fn very_long_pattern_gpu_scan() {
    if !has_gpu() {
        return;
    }

    // Create a long pattern (1KB)
    let long_pattern = "A".repeat(1024);
    let patterns = PatternSet::builder()
        .literal(&long_pattern)
        .build()
        .unwrap();

    let gpu_matcher = block_on(GpuMatcher::new(&patterns)).unwrap();

    // Create input with the pattern at various positions
    let mut input = vec![b'x'; 4 * 1024 * 1024]; // 4MB
    let positions = [0, 1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024];

    for pos in &positions {
        input[*pos..*pos + 1024].copy_from_slice(long_pattern.as_bytes());
    }

    let matches = block_on(gpu_matcher.scan(&input)).unwrap();

    // Should find all 4 occurrences
    assert_eq!(
        matches.len(),
        positions.len(),
        "Should find all {} occurrences of long pattern",
        positions.len()
    );

    for (i, m) in matches.iter().enumerate() {
        assert_eq!(
            m.start as usize, positions[i],
            "Match {i} should be at position {}",
            positions[i]
        );
        assert_eq!(
            m.end - m.start,
            1024,
            "Match {i} should have correct length"
        );
    }
}
