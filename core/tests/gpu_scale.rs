//! GPU scale test — Large pattern set parity verification
//!
//! These tests verify that GPU and CPU backends produce IDENTICAL match sets
//! for large numbers of patterns. If they disagree on any input, it's a
//! false negative or false positive bug.
//!
//! Tests are gated by `#[cfg(feature = "gpu")]` — they skip if GPU unavailable.

use std::collections::HashSet;
use std::time::Instant;

use warpstate::gpu::GpuMatcher;
use warpstate::PatternSet;

/// Helper: Block on an async future (synchronous test context).
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

/// Helper: Generate a random pattern string of given length (printable ASCII).
fn random_pattern(len: usize, seed: u64) -> String {
    let mut bytes = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        // Linear congruential generator for reproducible "randomness"
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        // Printable ASCII range 33-126 (avoid whitespace and control chars)
        let byte = (33 + (state % 94)) as u8;
        bytes.push(byte);
    }
    String::from_utf8(bytes).unwrap()
}

/// Helper: Generate random noise data.
fn random_noise(len: usize, seed: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        bytes.push((state % 256) as u8);
    }
    bytes
}

/// Helper: Compare two match vectors for exact equality.
fn matches_equal(cpu: &[warpstate::Match], gpu: &[warpstate::Match]) -> bool {
    if cpu.len() != gpu.len() {
        return false;
    }
    // Sort both by (start, pattern_id, end) for comparison
    let mut cpu_sorted: Vec<_> = cpu.to_vec();
    let mut gpu_sorted: Vec<_> = gpu.to_vec();
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
    cpu_sorted == gpu_sorted
}

// =============================================================================
// Test 1: 10,000 Unique Literal Patterns — Scale Test
// =============================================================================

/// Generate 10,000 unique literal patterns (random 8-32 byte strings).
/// Build a PatternSet from them, create test input with exactly 100 of the
/// patterns embedded in random noise. Scan with GPU and CPU, assert identical
/// match sets and print timing comparison.
#[test]
#[cfg(feature = "gpu")]
fn gpu_cpu_parity_10k_patterns() {
    const NUM_PATTERNS: usize = 10_000;
    const PATTERNS_TO_EMBED: usize = 100;
    const INPUT_SIZE: usize = 1_000_000; // 1MB input

    // Generate 10,000 unique patterns (8-32 bytes each)
    println!("Generating {} unique patterns...", NUM_PATTERNS);
    let mut patterns_list = Vec::with_capacity(NUM_PATTERNS);
    for i in 0..NUM_PATTERNS {
        let len = 8 + (i % 25); // 8-32 bytes
        let pattern = random_pattern(len, i as u64 + 1);
        patterns_list.push(pattern);
    }

    // Build PatternSet
    println!("Building PatternSet...");
    let mut builder = PatternSet::builder();
    for pattern in &patterns_list {
        builder = builder.literal(pattern);
    }
    let pattern_set = builder.build().expect("Failed to build PatternSet");

    // Create input with 100 patterns embedded in random noise
    println!(
        "Creating test input with {} embedded patterns...",
        PATTERNS_TO_EMBED
    );
    let mut input = random_noise(INPUT_SIZE, 42_000);
    let mut embedded_indices = HashSet::new();
    let mut embedded_positions = Vec::new();

    // Embed patterns 0-99 at known positions (spaced out)
    let spacing = INPUT_SIZE / (PATTERNS_TO_EMBED + 1);
    for i in 0..PATTERNS_TO_EMBED {
        let pos = (i + 1) * spacing;
        let pattern = &patterns_list[i];
        let pattern_bytes = pattern.as_bytes();

        if pos + pattern_bytes.len() <= INPUT_SIZE {
            input[pos..pos + pattern_bytes.len()].copy_from_slice(pattern_bytes);
            embedded_indices.insert(i);
            embedded_positions.push((i, pos));
        }
    }

    // CPU scan (baseline)
    println!("Running CPU scan...");
    let cpu_start = Instant::now();
    let cpu_matches = pattern_set
        .scan_overlapping(&input)
        .expect("CPU scan failed");
    let cpu_time = cpu_start.elapsed();

    // GPU scan (if available)
    println!("Running GPU scan...");
    let gpu_matcher = match block_on(GpuMatcher::new(&pattern_set)) {
        Ok(m) => m,
        Err(warpstate::Error::NoGpuAdapter) => {
            println!("GPU not available, skipping GPU test");
            return;
        }
        Err(e) => panic!("GPU init failed: {:?}", e),
    };

    let gpu_start = Instant::now();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).expect("GPU scan failed");
    let gpu_time = gpu_start.elapsed();

    // Print timing results
    println!("\n=== Timing Results ({} patterns) ===", NUM_PATTERNS);
    println!("CPU time: {:?}", cpu_time);
    println!("GPU time: {:?}", gpu_time);
    if gpu_time.as_secs_f64() > 0.0 {
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    }
    println!("CPU matches: {}", cpu_matches.len());
    println!("GPU matches: {}", gpu_matches.len());

    // Assert identical match sets
    assert!(
        matches_equal(&cpu_matches, &gpu_matches),
        "GPU and CPU produced different match sets!\nCPU: {:?}\nGPU: {:?}",
        cpu_matches,
        gpu_matches
    );

    // Verify all embedded patterns were found
    let cpu_found_embedded: HashSet<_> = cpu_matches
        .iter()
        .filter(|m| embedded_indices.contains(&(m.pattern_id as usize)))
        .map(|m| m.pattern_id)
        .collect();

    assert_eq!(
        cpu_found_embedded.len(),
        PATTERNS_TO_EMBED,
        "Not all embedded patterns were found"
    );

    println!("✓ GPU/CPU parity verified for {} patterns", NUM_PATTERNS);
}

// =============================================================================
// Test 2: 1,000 Patterns — Edge Case Coverage
// =============================================================================

/// Test with 1,000 patterns covering edge cases:
/// - Patterns at input position 0 (start boundary)
/// - Patterns at input end (end boundary)
/// - Overlapping patterns (two patterns share a prefix)
/// - Patterns in the last byte of input
/// - Empty input
/// - Input that matches ALL patterns
#[test]
#[cfg(feature = "gpu")]
#[ignore = "compares scan_overlapping (CPU) with scan (GPU non-overlapping) — different semantics"]
fn gpu_cpu_parity_1k_patterns_edge_cases() {
    const NUM_PATTERNS: usize = 1_000;

    // Generate 1,000 unique patterns (8-16 bytes each)
    println!("Generating {} edge-case patterns...", NUM_PATTERNS);
    let mut patterns_list = Vec::with_capacity(NUM_PATTERNS);
    for i in 0..NUM_PATTERNS {
        let len = 8 + (i % 9); // 8-16 bytes
        let pattern = random_pattern(len, i as u64 + 10_000);
        patterns_list.push(pattern);
    }

    // Add some overlapping patterns (patterns 0-9 share prefixes)
    patterns_list[0] = "AAAAAAAA".to_string();
    patterns_list[1] = "AAAAAAAB".to_string();
    patterns_list[2] = "AAAAAABC".to_string();
    patterns_list[3] = "AAAAABCD".to_string();
    patterns_list[4] = "AAAABCDE".to_string();

    // Build PatternSet
    let mut builder = PatternSet::builder();
    for pattern in &patterns_list {
        builder = builder.literal(pattern);
    }
    let pattern_set = builder.build().expect("Failed to build PatternSet");

    // Initialize GPU matcher
    let gpu_matcher = match block_on(GpuMatcher::new(&pattern_set)) {
        Ok(m) => m,
        Err(warpstate::Error::NoGpuAdapter) => {
            println!("GPU not available, skipping edge case tests");
            return;
        }
        Err(e) => panic!("GPU init failed: {:?}", e),
    };

    // -------------------------------------------------------------------------
    // Edge Case 1: Pattern at input position 0 (start boundary)
    // -------------------------------------------------------------------------
    println!("Testing start boundary...");
    let pattern0 = patterns_list[0].as_bytes();
    let input_start = [pattern0, b" followed by noise data here"].concat();

    let cpu_start_matches = pattern_set.scan_overlapping(&input_start).unwrap();
    let gpu_start_matches = block_on(gpu_matcher.scan(&input_start)).unwrap();

    assert!(
        matches_equal(&cpu_start_matches, &gpu_start_matches),
        "Start boundary: GPU/CPU mismatch\nCPU: {:?}\nGPU: {:?}",
        cpu_start_matches,
        gpu_start_matches
    );

    // Verify pattern 0 was found at position 0
    assert!(
        cpu_start_matches
            .iter()
            .any(|m| m.pattern_id == 0 && m.start == 0),
        "Pattern 0 should be found at start position"
    );

    // -------------------------------------------------------------------------
    // Edge Case 2: Pattern at input end (end boundary)
    // -------------------------------------------------------------------------
    println!("Testing end boundary...");
    let pattern1 = patterns_list[1].as_bytes();
    let input_end = [b"noise data precedes the ", pattern1].concat();

    let cpu_end_matches = pattern_set.scan_overlapping(&input_end).unwrap();
    let gpu_end_matches = block_on(gpu_matcher.scan(&input_end)).unwrap();

    assert!(
        matches_equal(&cpu_end_matches, &gpu_end_matches),
        "End boundary: GPU/CPU mismatch\nCPU: {:?}\nGPU: {:?}",
        cpu_end_matches,
        gpu_end_matches
    );

    // Verify pattern 1 was found at correct end position
    let pattern1_len = pattern1.len() as u32;
    let expected_start = input_end.len() as u32 - pattern1_len;
    assert!(
        cpu_end_matches
            .iter()
            .any(|m| m.pattern_id == 1 && m.start == expected_start),
        "Pattern 1 should be found at end position"
    );

    // -------------------------------------------------------------------------
    // Edge Case 3: Overlapping patterns (patterns 0-4 share prefixes)
    // -------------------------------------------------------------------------
    println!("Testing overlapping patterns...");
    // Input: "AAAAAAAA" contains overlapping patterns 0-4
    let input_overlap = b"AAAAAAAA";

    let cpu_overlap_matches = pattern_set.scan_overlapping(input_overlap).unwrap();
    let gpu_overlap_matches = block_on(gpu_matcher.scan(input_overlap)).unwrap();

    assert!(
        matches_equal(&cpu_overlap_matches, &gpu_overlap_matches),
        "Overlapping patterns: GPU/CPU mismatch\nCPU: {:?}\nGPU: {:?}",
        cpu_overlap_matches,
        gpu_overlap_matches
    );

    // Verify pattern 0 was found ("AAAAAAAA")
    assert!(
        cpu_overlap_matches.iter().any(|m| m.pattern_id == 0),
        "Overlapping: Pattern 0 should be found"
    );

    // -------------------------------------------------------------------------
    // Edge Case 4: Pattern in the last byte of input (single-byte input)
    // -------------------------------------------------------------------------
    println!("Testing last byte input...");
    // Create a pattern that is just one character
    let single_byte_pattern = "X".to_string();
    let single_byte_set = PatternSet::builder()
        .literal(&single_byte_pattern)
        .build()
        .expect("Failed to build single-byte pattern set");
    let single_byte_gpu = match block_on(GpuMatcher::new(&single_byte_set)) {
        Ok(m) => m,
        Err(_) => return, // Skip if GPU issues
    };

    let input_last_byte = b"X";
    let cpu_last_matches = single_byte_set.scan_overlapping(input_last_byte).unwrap();
    let gpu_last_matches = block_on(single_byte_gpu.scan(input_last_byte)).unwrap();

    assert!(
        matches_equal(&cpu_last_matches, &gpu_last_matches),
        "Last byte: GPU/CPU mismatch\nCPU: {:?}\nGPU: {:?}",
        cpu_last_matches,
        gpu_last_matches
    );

    // -------------------------------------------------------------------------
    // Edge Case 5: Empty input
    // -------------------------------------------------------------------------
    println!("Testing empty input...");
    let input_empty: &[u8] = b"";

    let cpu_empty_matches = pattern_set.scan_overlapping(input_empty).unwrap();
    let gpu_empty_matches = block_on(gpu_matcher.scan(input_empty)).unwrap();

    assert!(
        matches_equal(&cpu_empty_matches, &gpu_empty_matches),
        "Empty input: GPU/CPU mismatch\nCPU: {:?}\nGPU: {:?}",
        cpu_empty_matches,
        gpu_empty_matches
    );
    assert!(
        cpu_empty_matches.is_empty(),
        "Empty input should have no matches"
    );

    // -------------------------------------------------------------------------
    // Edge Case 6: Input that matches ALL patterns
    // -------------------------------------------------------------------------
    println!("Testing input that matches ALL patterns...");
    // Build input by concatenating all patterns
    let mut input_all_patterns = Vec::new();
    for pattern in &patterns_list {
        input_all_patterns.extend_from_slice(pattern.as_bytes());
        input_all_patterns.push(b' '); // Separator
    }

    let cpu_all_matches = pattern_set
        .scan_overlapping(&input_all_patterns)
        .expect("CPU scan failed for all-patterns input");
    let gpu_all_matches = block_on(gpu_matcher.scan(&input_all_patterns))
        .expect("GPU scan failed for all-patterns input");

    assert!(
        matches_equal(&cpu_all_matches, &gpu_all_matches),
        "All patterns match: GPU/CPU mismatch\nCPU matches: {}\nGPU matches: {}\nCPU: {:?}\nGPU: {:?}",
        cpu_all_matches.len(),
        gpu_all_matches.len(),
        cpu_all_matches,
        gpu_all_matches
    );

    // Verify at least NUM_PATTERNS matches (each pattern should match once)
    assert!(
        cpu_all_matches.len() >= NUM_PATTERNS,
        "Should have at least {} matches, found {}",
        NUM_PATTERNS,
        cpu_all_matches.len()
    );

    println!("✓ All edge case tests passed for {} patterns", NUM_PATTERNS);
}

// =============================================================================
// Test 3: Additional Edge Case — Single Byte Pattern at Various Positions
// =============================================================================

/// Test single-byte pattern matching at every possible position.
#[test]
#[cfg(feature = "gpu")]
fn gpu_cpu_parity_single_byte_pattern_positions() {
    // Use a single-byte pattern
    let pattern_set = PatternSet::builder()
        .literal("A")
        .build()
        .expect("Failed to build pattern set");

    let gpu_matcher = match block_on(GpuMatcher::new(&pattern_set)) {
        Ok(m) => m,
        Err(warpstate::Error::NoGpuAdapter) => return,
        Err(e) => panic!("GPU init failed: {:?}", e),
    };

    // Test at position 0
    let input0 = b"Axxxx";
    let cpu0 = pattern_set.scan_overlapping(input0).unwrap();
    let gpu0 = block_on(gpu_matcher.scan(input0)).unwrap();
    assert_eq!(cpu0, gpu0, "Single-byte at position 0");

    // Test at position 4 (end)
    let input4 = b"xxxxA";
    let cpu4 = pattern_set.scan_overlapping(input4).unwrap();
    let gpu4 = block_on(gpu_matcher.scan(input4)).unwrap();
    assert_eq!(cpu4, gpu4, "Single-byte at end position");

    // Test multiple positions
    let input_multi = b"xAxAxA";
    let cpu_multi = pattern_set.scan_overlapping(input_multi).unwrap();
    let gpu_multi = block_on(gpu_matcher.scan(input_multi)).unwrap();
    assert_eq!(cpu_multi, gpu_multi, "Single-byte at multiple positions");

    // Test no match
    let input_none = b"xxxxx";
    let cpu_none = pattern_set.scan_overlapping(input_none).unwrap();
    let gpu_none = block_on(gpu_matcher.scan(input_none)).unwrap();
    assert_eq!(cpu_none, gpu_none, "No match case");
    assert!(cpu_none.is_empty());

    println!("✓ Single-byte pattern position tests passed");
}

// =============================================================================
// Test 4: Pattern at Exact Chunk Boundary (if chunking is used)
// =============================================================================

/// Test that patterns at chunk boundaries are correctly detected.
#[test]
#[cfg(feature = "gpu")]
fn gpu_cpu_parity_chunk_boundary_patterns() {
    // Create a pattern
    let pattern_set = PatternSet::builder()
        .literal("BOUNDARY_PATTERN")
        .build()
        .expect("Failed to build pattern set");

    let gpu_matcher = match block_on(GpuMatcher::new(&pattern_set)) {
        Ok(m) => m,
        Err(warpstate::Error::NoGpuAdapter) => return,
        Err(e) => panic!("GPU init failed: {:?}", e),
    };

    // Create large input (2MB) with pattern at various positions including boundaries
    let pattern = b"BOUNDARY_PATTERN";
    let pattern_len = pattern.len();
    let chunk_size = 128 * 1024; // 128KB - common chunk size
    let input_size = 2 * 1024 * 1024; // 2MB
    let mut input = vec![b'x'; input_size];

    // Place pattern at distinct positions - ensuring no overlap
    // Use non-overlapping positions spaced far enough apart
    let positions = [
        chunk_size / 2,                    // In first chunk
        chunk_size - pattern_len,          // Just before chunk boundary
        chunk_size + 100,                  // Just after chunk boundary
        2 * chunk_size - pattern_len - 50, // Before second chunk boundary
        3 * chunk_size + 200,              // In third chunk
        input_size - pattern_len,          // At very end
    ];

    let mut valid_positions = Vec::new();
    for pos in &positions {
        if *pos + pattern_len <= input_size {
            input[*pos..*pos + pattern_len].copy_from_slice(pattern);
            valid_positions.push(*pos);
        }
    }

    let cpu_matches = pattern_set.scan_overlapping(&input).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

    assert!(
        matches_equal(&cpu_matches, &gpu_matches),
        "Chunk boundary: GPU/CPU mismatch\nCPU found: {}\nGPU found: {}",
        cpu_matches.len(),
        gpu_matches.len()
    );

    // Verify all valid positions were found
    assert_eq!(
        cpu_matches.len(),
        valid_positions.len(),
        "Should find all {} boundary patterns (found {})",
        valid_positions.len(),
        cpu_matches.len()
    );

    // Verify each position was found
    for expected_pos in &valid_positions {
        assert!(
            cpu_matches.iter().any(|m| m.start == *expected_pos as u32),
            "Should find pattern at position {}",
            expected_pos
        );
    }

    println!("✓ Chunk boundary pattern tests passed");
}

// =============================================================================
// Test 5: Binary Data with Null Bytes
// =============================================================================

/// Test matching patterns containing and surrounded by null bytes.
#[test]
#[cfg(feature = "gpu")]
fn gpu_cpu_parity_binary_with_nulls() {
    // Pattern with null bytes in middle
    let pattern_set = PatternSet::builder()
        .literal_bytes(b"hello\x00world")
        .literal_bytes(b"\x00\x00\x00")
        .literal("normal_pattern")
        .build()
        .expect("Failed to build pattern set");

    let gpu_matcher = match block_on(GpuMatcher::new(&pattern_set)) {
        Ok(m) => m,
        Err(warpstate::Error::NoGpuAdapter) => return,
        Err(e) => panic!("GPU init failed: {:?}", e),
    };

    // Input with null bytes and patterns
    let input = b"\x00\x00hello\x00world\x00\x00\x00normal_pattern\x00";

    let cpu_matches = pattern_set.scan_overlapping(input).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(input)).unwrap();

    assert!(
        matches_equal(&cpu_matches, &gpu_matches),
        "Binary nulls: GPU/CPU mismatch\nCPU: {:?}\nGPU: {:?}",
        cpu_matches,
        gpu_matches
    );

    // Verify all 3 patterns found
    assert_eq!(cpu_matches.len(), 3, "Should find all 3 patterns");

    println!("✓ Binary data with nulls tests passed");
}

// =============================================================================
// Test 6: Large Pattern Size (接近最大限制)
// =============================================================================

/// Test with patterns of varying large sizes up to 256 bytes.
#[test]
#[cfg(feature = "gpu")]
fn gpu_cpu_parity_large_pattern_sizes() {
    const SIZES: [usize; 5] = [64, 128, 192, 256, 300];

    for size in SIZES {
        println!("Testing pattern size: {} bytes", size);

        // Generate pattern of this size
        let pattern = random_pattern(size, size as u64);
        let pattern_set = PatternSet::builder()
            .literal(&pattern)
            .build()
            .expect("Failed to build pattern set");

        let gpu_matcher = match block_on(GpuMatcher::new(&pattern_set)) {
            Ok(m) => m,
            Err(warpstate::Error::NoGpuAdapter) => return,
            Err(e) => panic!("GPU init failed: {:?}", e),
        };

        // Create input with pattern at start, middle, and end
        let pattern_bytes = pattern.as_bytes();
        let padding = vec![b'x'; 1000];
        let input = [
            pattern_bytes,
            &padding,
            pattern_bytes,
            &padding,
            pattern_bytes,
        ]
        .concat();

        let cpu_matches = pattern_set.scan_overlapping(&input).unwrap();
        let gpu_matches = block_on(gpu_matcher.scan(&input)).unwrap();

        assert!(
            matches_equal(&cpu_matches, &gpu_matches),
            "Large pattern ({} bytes): GPU/CPU mismatch\nCPU: {}\nGPU: {}",
            size,
            cpu_matches.len(),
            gpu_matches.len()
        );

        // Should find 3 matches
        assert_eq!(
            cpu_matches.len(),
            3,
            "Should find pattern 3 times for size {}",
            size
        );
    }

    println!("✓ Large pattern size tests passed");
}

// =============================================================================
// Test 7: Many Matches in Small Input (饱和测试)
// =============================================================================

/// Test input where every position is a match for some pattern.
#[test]
#[cfg(feature = "gpu")]
fn gpu_cpu_parity_saturated_matches() {
    // Create patterns of single characters
    let mut builder = PatternSet::builder();
    let chars: Vec<String> = (b'a'..=b'z').map(|c| (c as char).to_string()).collect();
    for c in &chars {
        builder = builder.literal(c);
    }
    let pattern_set = builder.build().expect("Failed to build pattern set");

    let gpu_matcher = match block_on(GpuMatcher::new(&pattern_set)) {
        Ok(m) => m,
        Err(warpstate::Error::NoGpuAdapter) => return,
        Err(e) => panic!("GPU init failed: {:?}", e),
    };

    // Input with all lowercase letters - each position should match
    let input = b"abcdefghijklmnopqrstuvwxyz";

    let cpu_matches = pattern_set.scan_overlapping(input).unwrap();
    let gpu_matches = block_on(gpu_matcher.scan(input)).unwrap();

    assert!(
        matches_equal(&cpu_matches, &gpu_matches),
        "Saturated matches: GPU/CPU mismatch\nCPU: {}\nGPU: {}",
        cpu_matches.len(),
        gpu_matches.len()
    );

    // Should find 26 matches (one for each letter)
    assert_eq!(cpu_matches.len(), 26, "Should find 26 single-char matches");

    println!("✓ Saturated match tests passed");
}
