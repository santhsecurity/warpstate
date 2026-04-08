use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use warpstate::PatternSet;

fn bench_compact_dfa_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_dfa_throughput");
    let payload_size = 10 * 1024 * 1024; // 10 MB payload
    let payload = vec![b'a'; payload_size];

    for &pattern_count in &[1, 10, 100, 1000] {
        let mut builder = PatternSet::builder();
        let mut strings = Vec::new();
        for i in 0..pattern_count {
            strings.push(format!("b[0-9]+{i}"));
        }
        for s in &strings {
            builder = builder.regex(s.as_str());
        }
        let patterns = builder.build().unwrap();

        group.throughput(Throughput::Bytes(payload_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(pattern_count),
            &pattern_count,
            |b, _| {
                b.iter(|| {
                    black_box(patterns.scan(&payload).unwrap());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_compact_dfa_throughput);
criterion_main!(benches);
