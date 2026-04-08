//! GPU vs CPU performance benchmark for warpstate.
//!
//! Measures throughput (MB/s) and speedup ratio across different pattern counts.
//!
//! # Running on GPU-enabled machines
//!
//! Prerequisites:
//! 1. Vulkan drivers installed
//! 2. Run with: `cargo bench --bench gpu_vs_cpu`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use warpstate::PatternSet;

const SEED: u64 = 0xDEADBEEF_CAFE_BABE;
const INPUT_SIZE: usize = 1_048_576;
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
    builder.build().expect("valid pattern set")
}

fn generate_test_data(patterns: &[Vec<u8>]) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let mut data = random_bytes(&mut rng, INPUT_SIZE);

    for pattern in patterns.iter().take(50) {
        let max_offset = INPUT_SIZE.saturating_sub(pattern.len());
        if max_offset > 0 {
            let offset = rng.gen_range(0..=max_offset);
            data[offset..offset + pattern.len()].copy_from_slice(pattern);
        }
    }

    data
}

fn bench_cpu_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_throughput");
    group.throughput(Throughput::Bytes(INPUT_SIZE as u64));
    group.sample_size(10);

    for count in [1_000, 10_000, 100_000] {
        let raw_patterns = generate_patterns(count);
        let pattern_set = build_pattern_set(&raw_patterns);
        let test_data = generate_test_data(&raw_patterns);

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &(&pattern_set, test_data),
            |b, (ps, data)| {
                b.iter(|| black_box(ps.scan(data).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpu_throughput);
criterion_main!(benches);
