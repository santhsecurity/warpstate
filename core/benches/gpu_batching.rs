use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use warpstate::{
    batch::{scan_batch_gpu, ScanItem},
    AutoMatcher, PatternSet,
};

fn patterned_data(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| b"abcdefghijklmnopqrstuvwxyz0123456789"[i % 36])
        .collect()
}

fn bench_gpu_batch_coalescing(c: &mut Criterion) {
    // 1000 patterns
    let mut builder = PatternSet::builder();
    for i in 0..10_000 {
        builder = builder.literal(&format!("sig_{i:07}_secret"));
    }
    let patterns = builder.build().unwrap();

    // We want to test the latency of dispatching small chunks (batch coalescing)
    let chunk_size = 4096; // 4KB per item
    let chunk_data = patterned_data(chunk_size);

    // Try batch sizes: 100 items (400KB), 1000 items (4MB), 10_000 items (40MB)
    let mut group = c.benchmark_group("gpu_dispatch/batch_coalescing");

    if let Ok(matcher) = pollster::block_on(AutoMatcher::new(&patterns)) {
        for &batch_count in &[100usize, 1_000, 10_000] {
            let total_bytes = batch_count * chunk_size;
            group.throughput(Throughput::Bytes(total_bytes as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(batch_count),
                &batch_count,
                |b, &count| {
                    let items: Vec<ScanItem> = (0..count)
                        .map(|id| ScanItem {
                            id: id as u64,
                            data: &chunk_data,
                        })
                        .collect();

                    b.iter(|| {
                        black_box(
                            pollster::block_on(scan_batch_gpu(&matcher, items.iter().copied()))
                                .unwrap(),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_gpu_batch_coalescing);
criterion_main!(benches);
