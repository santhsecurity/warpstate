#![cfg(feature = "gpu")]
//! Tests for streaming and chunked execution.
//!
//! These tests define the DESIRED behavior for a legendary GPU engine:
//! 1. Ability to handle infinite streams without loading the whole file into RAM.
//! 2. Ability to carry DFA state across chunk boundaries.
//! 3. Ability to find matches that are perfectly split across a chunk boundary.

use warpstate::*;

#[test]
fn match_split_across_chunk_boundary() {
    // DESIRED BEHAVIOR:
    // If the GPU can only process 1024 bytes per chunk, what happens if a 10-byte pattern
    // starts at byte 1020 and ends at byte 1030?
    // It MUST be found. The DFA state must persist across the memory transfer boundary.

    let ps = PatternSet::builder().regex("legendary").build().unwrap();

    // Configure engine to strictly chunk exactly at 1024 bytes
    let config = AutoMatcherConfig::default()
        .gpu_max_input_size(1024)
        .chunk_size(1024)
        .chunk_overlap(16);

    let engine = match pollster::block_on(warpstate::GpuMatcher::with_config(&ps, config)) {
        Ok(e) => e,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected error: {other:?}"),
    };

    // Create exactly 2048 bytes of data
    let mut data = vec![b'.'; 2048];
    // Put "legendary" right over the boundary: bytes 1020 through 1028
    data[1020..1029].copy_from_slice(b"legendary");

    let stream = warpstate::StreamPipeline::new(engine, 16);
    let matches = pollster::block_on(stream.scan(&data)).unwrap();

    for (i, m) in matches.iter().enumerate() {
        eprintln!(
            "  match[{i}]: pattern_id={}, start={}, end={}",
            m.pattern_id, m.start, m.end
        );
    }

    assert_eq!(
        matches.len(),
        1,
        "Failed to find pattern perfectly split across chunk boundary"
    );
    assert_eq!(matches[0].start, 1020);
    assert_eq!(matches[0].end, 1029);
}

#[test]
fn input_larger_than_vram() {
    // DESIRED BEHAVIOR:
    // A user provides a 5GB slice or memory-mapped file.
    // `warpstate` should NOT return `InputTooLarge`. It should transparently batch the data
    // into VRAM-sized chunks and return the combined indices.

    let ps = PatternSet::builder().regex("target").build().unwrap();

    // Tiny VRAM config
    let config = AutoMatcherConfig::default()
        .gpu_max_input_size(4096)
        .chunk_overlap(16);
    let engine = match pollster::block_on(warpstate::GpuMatcher::with_config(&ps, config)) {
        Ok(e) => e,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected error: {other:?}"),
    };

    // 100 KB payload, cleanly exceeding the 4KB limit exactly 25 times.
    let mut data = vec![b'0'; 100_000];
    data[99_000..99_006].copy_from_slice(b"target");

    let stream = warpstate::StreamPipeline::new(engine, 16);
    let matches = pollster::block_on(stream.scan(&data)).unwrap();

    assert_eq!(
        matches.len(),
        1,
        "Failed to scan payload larger than VRAM limits natively."
    );
    assert_eq!(matches[0].start, 99_000);
}
