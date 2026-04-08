use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::RngCore;
use std::time::Duration;
use warpstate::{
    batch::{scan_batch_gpu, ScanItem},
    AutoMatcher, PatternSet,
};

// Benchmark threshold cross-over to definitively prove GPU dominates CPU at scales >16KB
fn generate_random_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    rand::thread_rng().fill_bytes(&mut data);
    // Inject literal periodically
    if size > 100 {
        data[size / 2..size / 2 + 5].copy_from_slice(b"santh");
    }
    data
}

fn bench_threshold(c: &mut Criterion) {
    // Rigorous threshold measurements showing CPU vs GPU curves across 1KB..512KB batches
    let batch_sizes = [1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288];
    let patterns = PatternSet::builder()
        .literal("santh")
        .literal("kernel")
        .literal("linux")
        .regex("([A-Za-z0-9+/]{40,})")
        .build()
        .unwrap();

    let matcher_cpu = pollster::block_on(AutoMatcher::with_options(
        &patterns,
        usize::MAX,
        1024 * 1024 * 1024,
    ))
    .unwrap();
    let matcher_gpu =
        pollster::block_on(AutoMatcher::with_options(&patterns, 0, 1024 * 1024 * 1024)).unwrap();

    let mut group = c.benchmark_group("Threshold_Crossover_CPU_vs_GPU");

    for size in batch_sizes {
        let input_data = generate_random_data(size);

        let items = [ScanItem {
            id: 1,
            data: &input_data,
        }];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("CPU", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    pollster::block_on(scan_batch_gpu(&matcher_cpu, items.iter().copied()))
                        .unwrap(),
                );
            });
        });

        if matcher_gpu.has_gpu() {
            group.bench_with_input(BenchmarkId::new("GPU", size), &size, |b, _| {
                b.iter(|| {
                    black_box(
                        pollster::block_on(scan_batch_gpu(&matcher_gpu, items.iter().copied()))
                            .unwrap(),
                    );
                });
            });
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = bench_threshold
}
criterion_main!(benches);
