//! Regression benchmarks for warpstate.
//!
//! Run: `cargo bench --bench regression`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::Rng;
use warpstate::PatternSet;

/// Generate random ASCII data of the given size.
fn random_data(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(b'a'..=b'z')).collect()
}

/// Generate N patterns of the given length, each random ASCII.
fn random_patterns(count: usize, len: usize) -> PatternSet {
    let mut rng = rand::thread_rng();
    let mut builder = PatternSet::builder();
    for _ in 0..count {
        let pat: String = (0..len)
            .map(|_| rng.gen_range(b'a'..=b'z') as char)
            .collect();
        builder = builder.literal(&pat);
    }
    builder.build().unwrap()
}

/// cpu_scan_1kb — PatternSet::scan on 1KB random data with 5 patterns
fn cpu_scan_1kb(c: &mut Criterion) {
    let patterns = random_patterns(5, 8);
    let data = random_data(1_024);
    c.bench_function("cpu_scan_1kb", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });
}

/// cpu_scan_1mb — PatternSet::scan on 1MB random data with 5 patterns
fn cpu_scan_1mb(c: &mut Criterion) {
    let patterns = random_patterns(5, 8);
    let data = random_data(1_048_576);
    let mut group = c.benchmark_group("cpu_scan_1mb");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("scan", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });
    group.finish();
}

/// cpu_scan_50_patterns — 50 patterns on 100KB data
fn cpu_scan_50_patterns(c: &mut Criterion) {
    let patterns = random_patterns(50, 8);
    let data = random_data(100 * 1_024);
    let mut group = c.benchmark_group("cpu_scan_50_patterns");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("scan", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });
    group.finish();
}

/// pattern_compilation — build PatternSet from 10 literal patterns
fn pattern_compilation(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let patterns: Vec<String> = (0..10)
        .map(|_| (0..8).map(|_| rng.gen_range(b'a'..=b'z') as char).collect())
        .collect();

    c.bench_function("pattern_compilation", |b| {
        b.iter(|| {
            let mut builder = PatternSet::builder();
            for pat in &patterns {
                builder = builder.literal(pat);
            }
            black_box(builder.build().unwrap());
        });
    });
}

/// dfa_build — DFA construction from 3 regex patterns
fn dfa_build(c: &mut Criterion) {
    let patterns = [
        r"[a-z]+@gmail\.com",
        r"https?://[a-z0-9.-]+",
        r"\d{4}-\d{2}-\d{2}",
    ];
    let ids = [0, 1, 2];

    c.bench_function("dfa_build", |b| {
        b.iter(|| {
            black_box(warpstate::dfa::RegexDFA::build(&patterns, &ids).unwrap());
        });
    });
}

criterion_group!(
    benches,
    cpu_scan_1kb,
    cpu_scan_1mb,
    cpu_scan_50_patterns,
    pattern_compilation,
    dfa_build,
);
criterion_main!(benches);
