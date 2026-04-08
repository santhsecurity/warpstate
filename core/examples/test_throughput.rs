//! Comprehensive throughput test for CPU vs GPU comparison.
//!
//! Run with: cargo run --example test_throughput --release
//!
//! This benchmark measures:
//! - CPU throughput with 1K, 10K, and 100K patterns
//! - 1MB input with 50 patterns embedded at random positions
//! - Pattern length: 8-32 bytes (random)
//!
//! GPU benchmarks require a machine with Vulkan-compatible GPU.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use warpstate::PatternSet;

#[cfg(feature = "gpu")]
use warpstate::{AutoMatcherConfig, GpuMatcher};

const SEED: u64 = 0xDEADBEEF_CAFE_BABE;
const INPUT_SIZE: usize = 1_048_576; // 1MB
const MIN_PATTERN_LEN: usize = 8;
const MAX_PATTERN_LEN: usize = 32;

fn random_bytes(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let charset: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./=";
    (0..len)
        .map(|_| charset[rng.gen_range(0..charset.len())])
        .collect()
}

fn random_pattern(rng: &mut StdRng) -> Vec<u8> {
    let len = rng.gen_range(MIN_PATTERN_LEN..=MAX_PATTERN_LEN);
    random_bytes(rng, len)
}

fn generate_patterns(count: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..count).map(|_| random_pattern(&mut rng)).collect()
}

fn build_pattern_set(patterns: &[Vec<u8>]) -> PatternSet {
    let mut builder = PatternSet::builder();
    for pattern in patterns {
        builder = builder.literal_bytes(pattern.clone());
    }
    builder.build().unwrap()
}

fn generate_test_data(patterns: &[Vec<u8>]) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let mut data = random_bytes(&mut rng, INPUT_SIZE);

    // Embed 50 patterns at random positions
    for pattern in patterns.iter().take(50) {
        let max_offset = INPUT_SIZE.saturating_sub(pattern.len());
        if max_offset > 0 {
            let offset = rng.gen_range(0..=max_offset);
            data[offset..offset + pattern.len()].copy_from_slice(pattern);
        }
    }

    data
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     warpstate CPU/GPU Throughput Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!(
        "  Input size:     {} bytes ({} MB)",
        INPUT_SIZE,
        INPUT_SIZE / 1_048_576
    );
    println!(
        "  Pattern length: {}-{} bytes",
        MIN_PATTERN_LEN, MAX_PATTERN_LEN
    );
    println!("  Embedded:       50 patterns at random positions");
    println!();

    let pattern_counts = [1_000, 10_000, 100_000];

    // Store results for summary
    let mut results: Vec<(usize, f64)> = Vec::new();

    for &count in &pattern_counts {
        let patterns = generate_patterns(count);
        let pattern_set = build_pattern_set(&patterns);
        let test_data = generate_test_data(&patterns);

        // Warmup
        for _ in 0..3 {
            let _ = pattern_set.scan(&test_data);
        }

        // Benchmark CPU
        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = pattern_set.scan(&test_data);
        }
        let elapsed = start.elapsed();

        let per_iter = elapsed.as_secs_f64() / iterations as f64;
        let throughput_mbps = (INPUT_SIZE as f64) / per_iter / 1_048_576.0;

        results.push((count, throughput_mbps));

        println!("Patterns: {:>6}", format_number(count));
        println!(
            "  CPU: {:>8.2} MB/s  ({:>6.2} ms/scan)",
            throughput_mbps,
            per_iter * 1000.0
        );

        // Try GPU if available
        #[cfg(feature = "gpu")]
        {
            match try_gpu_benchmark(&pattern_set, &test_data, per_iter) {
                Some((gpu_mbps, speedup)) => {
                    println!("  GPU: {:>8.2} MB/s  ({:.2}x speedup)", gpu_mbps, speedup);
                }
                None => {
                    println!("  GPU: Not available on this machine");
                }
            }
        }

        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════");
    println!("SUMMARY - CPU Throughput");
    println!("───────────────────────────────────────────────────────────────────");
    for (count, throughput) in &results {
        println!(
            "  {:>6} patterns: {:>8.2} MB/s",
            format_number(*count),
            throughput
        );
    }
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Scaling analysis
    if results.len() >= 2 {
        let base = results[0].1;
        println!(
            "Scaling Analysis (relative to {} patterns):",
            format_number(results[0].0)
        );
        for (i, (count, throughput)) in results.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let ratio = throughput / base;
            println!(
                "  {} patterns: {:.2}x the patterns, {:.2}x the throughput",
                format_number(*count),
                *count as f64 / results[0].0 as f64,
                ratio
            );
        }
    }

    println!();
    println!("To run GPU benchmarks:");
    println!("  1. Run on a machine with GPU (e.g., santhserver with RTX 3080 Ti)");
    println!("  2. Install Vulkan drivers: sudo apt install nvidia-utils-xxx");
    println!("  3. Verify: vulkaninfo | grep deviceName");
}

#[cfg(feature = "gpu")]
fn try_gpu_benchmark(
    pattern_set: &PatternSet,
    test_data: &[u8],
    cpu_per_iter: f64,
) -> Option<(f64, f64)> {
    use std::panic::catch_unwind;

    let gpu_result = catch_unwind(|| {
        pollster::block_on(async {
            GpuMatcher::with_config(
                pattern_set,
                AutoMatcherConfig::new()
                    .chunk_size(INPUT_SIZE)
                    .gpu_max_input_size(INPUT_SIZE)
                    .max_matches(131_072),
            )
            .await
        })
    });

    match gpu_result {
        Ok(Ok(gpu_matcher)) => {
            // Warmup
            for _ in 0..3 {
                let _ = pollster::block_on(gpu_matcher.scan(test_data));
            }

            // Benchmark
            let iterations = 20;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = pollster::block_on(gpu_matcher.scan(test_data));
            }
            let elapsed = start.elapsed();

            let gpu_per_iter = elapsed.as_secs_f64() / iterations as f64;
            let gpu_throughput = (INPUT_SIZE as f64) / gpu_per_iter / 1_048_576.0;
            let speedup = cpu_per_iter / gpu_per_iter;

            Some((gpu_throughput, speedup))
        }
        _ => None,
    }
}
