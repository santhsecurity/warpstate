//! Throughput benchmarks for warpstate — REAL measurements for competitive analysis.
//!
//! Run: `cargo bench --bench throughput 2>&1`
//!
//! Measures:
//! - 100 literal patterns, 1MB input → MB/s
//! - 1000 literal patterns, 1MB input → MB/s
//! - 10000 literal patterns, 1MB input → MB/s
//! - 1 regex pattern, 1MB input → MB/s
//! - Comparison with grep command

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use std::process::Command;
use std::time::Instant;

use warpstate::{AutoMatcherConfig, GpuMatcher, PatternSet};

const INPUT_SIZE: usize = 1_048_576; // 1MB
const INPUT_SIZE_MB: f64 = 1.0;

/// Generate random lowercase ASCII data.
fn random_data(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(b'a'..=b'z')).collect()
}

/// Generate N random patterns of given length.
fn random_patterns(count: usize, len: usize) -> PatternSet {
    let mut rng = rand::thread_rng();
    let mut builder = PatternSet::builder();
    for _i in 0..count {
        let pat: String = (0..len)
            .map(|_| rng.gen_range(b'a'..=b'z') as char)
            .collect();
        builder = builder.literal(&pat);
    }
    builder.build().unwrap()
}

/// Generate one regex pattern.
fn single_regex_pattern() -> PatternSet {
    PatternSet::builder()
        .regex(r"[a-z]{5}[0-9]{3}_[a-z]+")
        .build()
        .unwrap()
}

/// Benchmark 100 literal patterns on 1MB input.
fn bench_100_literals(c: &mut Criterion) {
    // Print comparison at start of first benchmark
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(print_grep_comparison);

    let data = random_data(INPUT_SIZE);
    let patterns = random_patterns(100, 8);

    let mut group = c.benchmark_group("throughput/100_literals_1mb");
    group.throughput(Throughput::Bytes(INPUT_SIZE as u64));
    group.sample_size(20);

    // CPU scan
    group.bench_function("cpu", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });

    // GPU scan (if available)
    if let Ok(gpu) = pollster::block_on(GpuMatcher::new(&patterns)) {
        group.bench_function("gpu", |b| {
            b.iter(|| {
                black_box(pollster::block_on(gpu.scan(&data)).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark 1000 literal patterns on 1MB input.
fn bench_1000_literals(c: &mut Criterion) {
    let data = random_data(INPUT_SIZE);
    let patterns = random_patterns(1000, 8);

    let mut group = c.benchmark_group("throughput/1000_literals_1mb");
    group.throughput(Throughput::Bytes(INPUT_SIZE as u64));
    group.sample_size(20);

    // CPU scan
    group.bench_function("cpu", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });

    // GPU scan (if available)
    if let Ok(gpu) = pollster::block_on(GpuMatcher::new(&patterns)) {
        group.bench_function("gpu", |b| {
            b.iter(|| {
                black_box(pollster::block_on(gpu.scan(&data)).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark 10000 literal patterns on 1MB input.
fn bench_10000_literals(c: &mut Criterion) {
    let data = random_data(INPUT_SIZE);
    let patterns = random_patterns(10000, 8);

    let mut group = c.benchmark_group("throughput/10000_literals_1mb");
    group.throughput(Throughput::Bytes(INPUT_SIZE as u64));
    group.sample_size(10);

    // CPU scan
    group.bench_function("cpu", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });

    // GPU scan (if available)
    if let Ok(gpu) = pollster::block_on(GpuMatcher::new(&patterns)) {
        group.bench_function("gpu", |b| {
            b.iter(|| {
                black_box(pollster::block_on(gpu.scan(&data)).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark 1 regex pattern on 1MB input.
fn bench_1_regex(c: &mut Criterion) {
    let data = random_data(INPUT_SIZE);
    let patterns = single_regex_pattern();

    let mut group = c.benchmark_group("throughput/1_regex_1mb");
    group.throughput(Throughput::Bytes(INPUT_SIZE as u64));
    group.sample_size(20);

    // CPU scan
    group.bench_function("cpu", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });

    // GPU scan (if available)
    if let Ok(gpu) = pollster::block_on(GpuMatcher::new(&patterns)) {
        group.bench_function("gpu", |b| {
            b.iter(|| {
                black_box(pollster::block_on(gpu.scan(&data)).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark varying pattern counts to show scaling.
fn bench_scaling(c: &mut Criterion) {
    let data = random_data(INPUT_SIZE);
    let counts = [100usize, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("throughput/scaling_cpu");
    group.throughput(Throughput::Bytes(INPUT_SIZE as u64));

    for &count in &counts {
        let patterns = random_patterns(count, 8);
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &patterns,
            |b, patterns| {
                b.iter(|| {
                    black_box(patterns.scan(&data).unwrap());
                });
            },
        );
    }

    group.finish();

    // GPU scaling (separate group to handle GPU availability)
    if let Ok(_) = pollster::block_on(GpuMatcher::with_config(
        &random_patterns(100, 8),
        AutoMatcherConfig::new(),
    )) {
        let mut group = c.benchmark_group("throughput/scaling_gpu");
        group.throughput(Throughput::Bytes(INPUT_SIZE as u64));

        for &count in &counts {
            let patterns = random_patterns(count, 8);
            if let Ok(gpu) = pollster::block_on(GpuMatcher::new(&patterns)) {
                group.bench_with_input(BenchmarkId::from_parameter(count), &gpu, |b, gpu| {
                    b.iter(|| {
                        black_box(pollster::block_on(gpu.scan(&data)).unwrap());
                    });
                });
            }
        }

        group.finish();
    }
}

/// Benchmark auto-routing between CPU and GPU.
fn bench_auto_routing(c: &mut Criterion) {
    use warpstate::AutoMatcher;

    let patterns = random_patterns(1000, 8);

    let mut group = c.benchmark_group("throughput/auto_routing");

    // Small input (< threshold)
    let small_data = random_data(1024); // 1KB
    group.throughput(Throughput::Bytes(small_data.len() as u64));

    if let Ok(matcher) = pollster::block_on(AutoMatcher::new(&patterns)) {
        group.bench_function("1kb_input", |b| {
            b.iter(|| {
                black_box(pollster::block_on(matcher.scan(&small_data)).unwrap());
            });
        });
    }

    // Large input (> threshold)
    let large_data = random_data(INPUT_SIZE);
    group.throughput(Throughput::Bytes(large_data.len() as u64));

    if let Ok(matcher) = pollster::block_on(AutoMatcher::new(&patterns)) {
        group.bench_function("1mb_input", |b| {
            b.iter(|| {
                black_box(pollster::block_on(matcher.scan(&large_data)).unwrap());
            });
        });
    }

    group.finish();
}

/// Run external grep benchmark for comparison.
fn run_grep_benchmark() -> Option<f64> {
    let test_file = "/tmp/warpstate_benchmark_1mb.bin";

    // Create test file
    let data = random_data(INPUT_SIZE);
    std::fs::write(test_file, &data).ok()?;

    // Choose a pattern that won't match (worst case for grep)
    let pattern = "xyzxyzxyz";

    // Warmup
    let _ = Command::new("grep")
        .args(["-c", pattern, test_file])
        .output();

    // Benchmark
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Command::new("grep")
            .args(["-c", pattern, test_file])
            .output();
    }
    let elapsed = start.elapsed();

    // Cleanup
    let _ = std::fs::remove_file(test_file);

    let total_mb = INPUT_SIZE_MB * iterations as f64;
    let seconds = elapsed.as_secs_f64();
    let throughput = total_mb / seconds;

    Some(throughput)
}

/// Print comparison with grep.
fn print_grep_comparison() {
    println!("\n{:=<70}", "");
    println!("EXTERNAL TOOL COMPARISON");
    println!("{:=<70}", "");

    match run_grep_benchmark() {
        Some(grep_throughput) => {
            println!(
                "grep -c (10 iterations on 1MB file): {:.2} MB/s",
                grep_throughput
            );
            println!("Note: grep is single-pattern, single-threaded for this comparison");
        }
        None => {
            println!("grep benchmark failed (grep may not be installed)");
        }
    }

    println!("{:=<70}\n", "");
}

/// Custom criterion configuration for stable measurements.
fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_100_literals, bench_1000_literals, bench_10000_literals, bench_1_regex, bench_scaling, bench_auto_routing
}

criterion_main!(benches);
