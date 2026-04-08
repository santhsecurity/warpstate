//! CPU vs GPU benchmark matrix for warpstate.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use warpstate::{AutoMatcherConfig, GpuMatcher, PatternSet};

fn patterned_data(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| b"abcdefghijklmnopqrstuvwxyz0123456789"[i % 36])
        .collect()
}

fn generated_patterns(count: usize) -> PatternSet {
    let mut builder = PatternSet::builder();
    for i in 0..count {
        builder = builder.literal(&format!("sig_{i:07}_secret"));
    }
    builder.build().unwrap()
}

fn bench_cpu_vs_gpu(c: &mut Criterion) {
    let data = patterned_data(8 * 1024 * 1024);
    let mut group = c.benchmark_group("cpu_vs_gpu/pattern_scale");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(data.len() as u64));

    for &count in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let patterns = generated_patterns(count);
        group.bench_with_input(BenchmarkId::new("cpu", count), &patterns, |b, patterns| {
            b.iter(|| {
                black_box(patterns.scan(&data).unwrap());
            });
        });

        if let Ok(gpu) = pollster::block_on(GpuMatcher::with_config(
            &patterns,
            AutoMatcherConfig::new()
                .chunk_size(8 * 1024 * 1024)
                .gpu_max_input_size(8 * 1024 * 1024)
                .max_matches(131_072),
        )) {
            group.bench_with_input(BenchmarkId::new("gpu", count), &gpu, |b, gpu| {
                b.iter(|| {
                    black_box(pollster::block_on(gpu.scan(&data)).unwrap());
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_cpu_vs_gpu);
criterion_main!(benches);
