//! Criterion benchmarks for warpstate.
//!
//! These benchmarks measure:
//! - Config construction overhead
//! - Pattern compilation (DFA and literal)
//! - Buffer pool hit/miss performance
//! - CPU vs GPU scan performance

use aho_corasick::AhoCorasick;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use warpstate::{
    scan_aho_corasick, AutoMatcherConfig, GpuMatcher, HashScanner, PatternIR, PatternSet,
};

const HASH_WINDOW_LEN: usize = 8;

#[derive(Clone)]
struct BenchPrefilterTable {
    prefix_meta: Vec<[u32; 4]>,
    bucket_ranges: Vec<[u32; 2]>,
    entries: Vec<[u32; 2]>,
}

impl BenchPrefilterTable {
    fn build(ir: &PatternIR) -> Self {
        let mut prefix_groups = vec![Vec::<[u32; 2]>::new(); HASH_WINDOW_LEN];
        for (literal_index, &(_, len)) in ir.offsets.iter().enumerate() {
            let start = ir.offsets[literal_index].0 as usize;
            let len = len as usize;
            let hash = fnv1a_hash(&ir.packed_bytes[start..start + len], HASH_WINDOW_LEN);
            let prefix_len = len.clamp(1, HASH_WINDOW_LEN);
            prefix_groups[prefix_len - 1].push([hash, literal_index as u32]);
        }

        let mut prefix_meta = vec![[0u32; 4]; HASH_WINDOW_LEN];
        let mut bucket_ranges = Vec::new();
        let mut entries = Vec::with_capacity(ir.offsets.len());
        for (prefix_idx, group) in prefix_groups.into_iter().enumerate() {
            if group.is_empty() {
                continue;
            }

            let bucket_count = (group.len() * 2).next_power_of_two();
            let bucket_offset = bucket_ranges.len() as u32;
            let mut buckets = vec![Vec::<[u32; 2]>::new(); bucket_count];
            for entry in group {
                let bucket_idx = (entry[0] as usize) & (bucket_count - 1);
                buckets[bucket_idx].push(entry);
            }

            for bucket in buckets {
                let entry_start = entries.len() as u32;
                let entry_count = bucket.len() as u32;
                bucket_ranges.push([entry_start, entry_count]);
                entries.extend(bucket);
            }

            prefix_meta[prefix_idx] = [
                bucket_offset,
                bucket_count as u32 - 1,
                bucket_count as u32,
                0,
            ];
        }

        Self {
            prefix_meta,
            bucket_ranges,
            entries,
        }
    }
}

fn fnv1a_hash(bytes: &[u8], window_len: usize) -> u32 {
    let mut hash = 2_166_136_261u32;
    for &byte in bytes.iter().take(window_len.max(1)) {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

fn ir_literal_slices(ir: &PatternIR) -> Vec<&[u8]> {
    ir.offsets
        .iter()
        .map(|&(start, len)| &ir.packed_bytes[start as usize..(start + len) as usize])
        .collect()
}

fn linear_prefilter_scan(ir: &PatternIR, data: &[u8]) -> usize {
    let hashes: Vec<u32> = ir
        .offsets
        .iter()
        .map(|&(start, len)| {
            fnv1a_hash(
                &ir.packed_bytes[start as usize..(start + len) as usize],
                HASH_WINDOW_LEN,
            )
        })
        .collect();
    let mut candidate_count = 0usize;

    for pos in 0..data.len() {
        let mut is_candidate = false;
        for (literal_index, &(_, len)) in ir.offsets.iter().enumerate() {
            let pat_len = len as usize;
            if pos + pat_len > data.len() {
                continue;
            }
            let hash = fnv1a_hash(
                &data[pos..pos + pat_len.min(HASH_WINDOW_LEN)],
                HASH_WINDOW_LEN,
            );
            if hash == hashes[literal_index] {
                is_candidate = true;
                break;
            }
        }
        candidate_count += usize::from(is_candidate);
    }

    candidate_count
}

fn bucketed_prefilter_scan(ir: &PatternIR, table: &BenchPrefilterTable, data: &[u8]) -> usize {
    let mut candidate_count = 0usize;

    for pos in 0..data.len() {
        let mut hash = 2_166_136_261u32;
        let max_probe_len = HASH_WINDOW_LEN.min(data.len() - pos);
        let mut is_candidate = false;

        for prefix_idx in 0..max_probe_len {
            hash ^= u32::from(data[pos + prefix_idx]);
            hash = hash.wrapping_mul(16_777_619);

            let meta = table.prefix_meta[prefix_idx];
            if meta[2] == 0 {
                continue;
            }

            let bucket_idx = (meta[0] + (hash & meta[1])) as usize;
            let [entry_start, entry_count] = table.bucket_ranges[bucket_idx];
            for entry in &table.entries[entry_start as usize..(entry_start + entry_count) as usize]
            {
                if entry[0] != hash {
                    continue;
                }

                let pat_len = ir.offsets[entry[1] as usize].1 as usize;
                if pos + pat_len <= data.len() {
                    is_candidate = true;
                    break;
                }
            }

            if is_candidate {
                break;
            }
        }

        candidate_count += usize::from(is_candidate);
    }

    candidate_count
}

fn make_literal_set(count: usize) -> PatternSet {
    (0..count)
        .fold(PatternSet::builder(), |builder, i| {
            let literal = format!("pattern_{i:05}_literal");
            builder.literal(&literal)
        })
        .build()
        .unwrap()
}

fn bench_hash_scanner_vs_aho(c: &mut Criterion) {
    let mut group = c.benchmark_group("literal_hash_scanner_vs_aho");
    group.sample_size(10);

    let data = vec![b'x'; 1_048_576];
    group.throughput(Throughput::Bytes(data.len() as u64));

    for count in [100usize, 10_000, 25_000] {
        let set = make_literal_set(count);
        let ir = set.ir();
        let aho_patterns = ir_literal_slices(&ir);
        let ac = AhoCorasick::new(&aho_patterns[..]).unwrap();
        let hash_scanner = HashScanner::build(&ir);

        group.bench_with_input(
            BenchmarkId::new("aho_build", count),
            &aho_patterns,
            |b, patterns| {
                b.iter(|| {
                    black_box(AhoCorasick::new(patterns.as_slice()).unwrap());
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("hash_build", count), &ir, |b, ir| {
            b.iter(|| {
                black_box(HashScanner::build(ir));
            });
        });

        group.bench_with_input(BenchmarkId::new("aho_scan", count), &ac, |b, ac| {
            b.iter(|| {
                let mut out_matches = [warpstate::Match::from_parts(0, 0, 0); 1000];
                black_box(scan_aho_corasick(ac, &data, &mut out_matches).unwrap());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("hash_scan", count),
            &hash_scanner,
            |b, scanner| {
                b.iter(|| {
                    black_box(scanner.scan(&data));
                });
            },
        );
    }

    group.finish();
}

fn bench_literal_prefilter_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("literal_prefilter_lookup");
    let data = vec![b'x'; 1024 * 1024];

    for count in [100usize, 1000, 10_000] {
        let set = make_literal_set(count);
        let ir = set.ir();
        let table = BenchPrefilterTable::build(ir);

        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("before_linear_scan", count),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(linear_prefilter_scan(ir, data));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("after_bucket_probe", count),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(bucketed_prefilter_scan(ir, &table, data));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark config construction with various settings
fn bench_config_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("config");

    group.bench_function("default", |b| {
        b.iter(|| {
            black_box(AutoMatcherConfig::default());
        });
    });

    group.bench_function("new_builder", |b| {
        b.iter(|| {
            black_box(AutoMatcherConfig::new());
        });
    });

    group.bench_function("full_custom", |b| {
        b.iter(|| {
            black_box(
                AutoMatcherConfig::new()
                    .gpu_threshold(1024)
                    .gpu_max_input_size(2048)
                    .max_matches(4096)
                    .chunk_size(8192)
                    .chunk_overlap(128)
                    .max_scan_depth(Some(100)),
            );
        });
    });

    group.finish();
}

/// Benchmark pattern compilation for various pattern types
fn bench_pattern_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_compile");

    // Single literal pattern
    group.bench_function("single_literal", |b| {
        b.iter(|| {
            black_box(PatternSet::builder().literal("password").build().unwrap());
        });
    });

    // Multiple literal patterns
    group.bench_function("ten_literals", |b| {
        b.iter(|| {
            black_box(
                PatternSet::builder()
                    .literal("password")
                    .literal("secret")
                    .literal("key")
                    .literal("token")
                    .literal("auth")
                    .literal("login")
                    .literal("admin")
                    .literal("root")
                    .literal("user")
                    .literal("pass")
                    .build()
                    .unwrap(),
            );
        });
    });

    // Simple regex
    group.bench_function("simple_regex", |b| {
        b.iter(|| {
            black_box(PatternSet::builder().regex("abc").build().unwrap());
        });
    });

    // Complex regex
    group.bench_function("complex_regex", |b| {
        b.iter(|| {
            black_box(
                PatternSet::builder()
                    .regex(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\(")
                    .build()
                    .unwrap(),
            );
        });
    });

    // Mixed patterns
    group.bench_function("mixed_patterns", |b| {
        b.iter(|| {
            black_box(
                PatternSet::builder()
                    .literal("password")
                    .regex(r"secret[_-]?key")
                    .literal("token")
                    .regex(r"api[_-]?key")
                    .build()
                    .unwrap(),
            );
        });
    });

    group.finish();
}

/// Benchmark buffer pool operations (hit vs miss)
fn bench_buffer_pool(c: &mut Criterion) {
    use warpstate::gpu::GpuMatcher;

    let mut group = c.benchmark_group("buffer_pool");

    // This benchmark requires a GPU, so we skip if not available
    let patterns = PatternSet::builder().literal("test").build().unwrap();

    if let Ok(gpu) = pollster::block_on(GpuMatcher::new(&patterns)) {
        // First scan populates the pool
        let _ = pollster::block_on(gpu.scan(b"test data for buffer pool"));

        group.bench_function("scan_with_pool_hit", |b| {
            b.iter(|| {
                black_box(pollster::block_on(gpu.scan(b"test data")).unwrap());
            });
        });

        // Scan with larger data (pool miss, needs new allocation)
        let large_data = vec![b'x'; 1024 * 1024];
        group.bench_with_input(
            BenchmarkId::new("scan_with_pool_miss", "1mb"),
            &large_data,
            |b, data| {
                b.iter(|| {
                    black_box(pollster::block_on(gpu.scan(data)).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CPU scanning at various input sizes
fn bench_cpu_scan(c: &mut Criterion) {
    let patterns = PatternSet::builder().literal("needle").build().unwrap();

    let mut group = c.benchmark_group("cpu_scan");

    for size in [1024, 64 * 1024, 1024 * 1024] {
        let data = vec![b'a'; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                black_box(patterns.scan(data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark GPU scanning at various input sizes (if GPU available)
fn bench_gpu_scan(c: &mut Criterion) {
    let patterns = PatternSet::builder().literal("needle").build().unwrap();

    let mut group = c.benchmark_group("gpu_scan");

    if let Ok(gpu) = pollster::block_on(GpuMatcher::new(&patterns)) {
        for size in [1024, 64 * 1024, 1024 * 1024] {
            let data = vec![b'a'; size];
            group.throughput(Throughput::Bytes(size as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
                b.iter(|| {
                    black_box(pollster::block_on(gpu.scan(data)).unwrap());
                });
            });
        }
    }

    group.finish();
}

/// Benchmark pattern matching with many patterns
fn bench_many_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("many_patterns");

    // Create pattern sets of various sizes
    for count in [10, 100, 1000] {
        let patterns: Vec<String> = (0..count)
            .map(|i| format!("pattern_{:04}_data", i))
            .collect();

        let set = patterns
            .iter()
            .fold(PatternSet::builder(), |b, p| b.literal(p))
            .build()
            .unwrap();

        let data = format!("test pattern_{:04}_data here", count / 2);
        group.bench_with_input(BenchmarkId::new("cpu_scan", count), &data, |b, data| {
            b.iter(|| {
                black_box(set.scan(data.as_bytes()).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_config_construction,
    bench_hash_scanner_vs_aho,
    bench_pattern_compile,
    bench_buffer_pool,
    bench_cpu_scan,
    bench_gpu_scan,
    bench_many_patterns,
    bench_literal_prefilter_lookup
);
criterion_main!(benches);
