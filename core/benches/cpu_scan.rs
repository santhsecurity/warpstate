//! CPU scan benchmarks for warpstate.
//!
//! Measures multi-pattern matching performance across:
//! - Different input sizes (1KB → 10MB)
//! - Different pattern counts (10 → 10,000)
//!
//! Run: `cargo bench --bench cpu_scan`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use warpstate::{dfa::MASK_STATE, PatternSet, PatternSetBuilder};

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

/// Benchmark: vary input size with fixed pattern count.
fn bench_input_sizes(c: &mut Criterion) {
    let patterns = random_patterns(100, 8);
    let mut group = c.benchmark_group("cpu_scan/input_size");

    for size in [1_024, 10_240, 102_400, 1_048_576, 10_485_760] {
        let data = random_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(patterns.scan(data).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: vary pattern count with fixed input size.
fn bench_pattern_counts(c: &mut Criterion) {
    let data = random_data(1_048_576); // 1MB
    let mut group = c.benchmark_group("cpu_scan/pattern_count");
    group.throughput(Throughput::Bytes(data.len() as u64));

    for count in [10, 100, 1_000, 5_000, 10_000] {
        let patterns = random_patterns(count, 8);
        group.bench_with_input(BenchmarkId::from_parameter(count), &data, |b, data| {
            b.iter(|| {
                black_box(patterns.scan(data).unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark: security use case — scan for many known-bad patterns in a large response.
fn bench_security_scan(c: &mut Criterion) {
    // Simulate scanning a 5MB HTTP response body for 500 attack signatures
    let patterns = random_patterns(500, 12);
    let data = random_data(5_242_880);

    c.bench_function("security_scan/5MB_500patterns", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });
}

/// Benchmark: real-world patterns (common strings, not random).
fn bench_realistic_patterns(c: &mut Criterion) {
    let mut builder = PatternSet::builder();
    let keywords = [
        "password",
        "secret",
        "api_key",
        "token",
        "authorization",
        "private_key",
        "access_key",
        "client_secret",
        "database_url",
        "aws_secret",
        "ssh-rsa",
        "BEGIN RSA PRIVATE KEY",
        "ghp_",
        "sk-",
        "Bearer ",
        "Basic ",
        "AKIA",
        "mysql://",
        "postgres://",
        "mongodb://",
    ];
    for kw in &keywords {
        builder = builder.literal(*kw);
    }
    let patterns = builder.build().unwrap();

    let data = random_data(1_048_576);

    c.bench_function("realistic/1MB_20_secret_patterns", |b| {
        b.iter(|| {
            black_box(patterns.scan(&data).unwrap());
        });
    });
}

/// Benchmark: 100MB input with 1, 10, and 100 patterns.
///
/// The payload embeds one guaranteed hit per KiB so match-heavy scans exercise
/// allocation growth and post-processing behavior in the CPU hot path.
fn bench_pattern_scale_100mb(c: &mut Criterion) {
    const FILE_SIZE: usize = 100 * 1_048_576;
    const STRIDE: usize = 1_024;
    const NEEDLE_LEN: usize = 9;

    fn make_pattern(index: usize) -> String {
        format!("p{index:03}match")
    }

    fn make_data(needle: &[u8]) -> Vec<u8> {
        let mut data = vec![b'x'; FILE_SIZE];
        let last_start = FILE_SIZE.saturating_sub(needle.len());
        for offset in (0..=last_start).step_by(STRIDE) {
            data[offset..offset + needle.len()].copy_from_slice(needle);
        }
        data
    }

    let anchor = make_pattern(0);
    assert_eq!(anchor.len(), NEEDLE_LEN);
    let data = make_data(anchor.as_bytes());

    let mut group = c.benchmark_group("cpu_scan/pattern_scale_100mb");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(FILE_SIZE as u64));

    for count in [1, 10, 100] {
        let mut builder = PatternSet::builder();
        for index in 0..count {
            builder = builder.literal(&make_pattern(index));
        }
        let patterns = builder.build().unwrap();

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
}

fn bench_regex_dfa_prefetch_100mb(c: &mut Criterion) {
    const FILE_SIZE: usize = 100 * 1_048_576;
    const TARGET_MIN_STATES: usize = 1_000;
    const PATTERN_COUNT: usize = 256;

    fn exact_regex_pattern(index: usize) -> String {
        format!("p{index:04}_[ab]cdefghijklmnopqrstuv")
    }

    fn build_large_regex_dfa() -> warpstate::PatternSet {
        let mut builder = PatternSetBuilder::default();
        for index in 0..PATTERN_COUNT {
            let pattern = exact_regex_pattern(index);
            builder = builder.regex(&pattern);
        }
        let pattern_set = builder.build().unwrap();
        assert!(
            !pattern_set.ir().regex_dfas().is_empty(),
            "expected regex patterns to produce a RegexDFA benchmark workload"
        );
        let dfa = &pattern_set.ir().regex_dfas()[0];
        let state_count = dfa.transition_table().len() / dfa.class_count() as usize;
        assert!(
            state_count >= TARGET_MIN_STATES,
            "expected at least {TARGET_MIN_STATES} DFA states, got {state_count}"
        );
        pattern_set
    }

    #[inline(always)]
    fn prefetch_transition(table: &[u32], idx: usize) {
        let ptr = unsafe { table.as_ptr().add(idx) }.cast::<u8>();

        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), core::arch::x86_64::_MM_HINT_T1);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::aarch64::__prefetch(ptr.cast(), 0, 3);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = ptr;
        }
    }

    #[inline(never)]
    fn walk_dfa(table_prefetch: bool, dfa: &warpstate::dfa::RegexDFA, haystack: &[u8]) -> u32 {
        let table = dfa.transition_table();
        let classes = dfa.byte_classes();
        let class_count = dfa.class_count() as usize;
        let mut state = dfa.start_state();

        for (pos, &byte) in haystack.iter().enumerate() {
            let next_byte = haystack.get(pos + 1).copied();
            let class_id = classes[usize::from(byte)] as usize;
            let state_idx = (state & MASK_STATE) as usize;
            state = table[state_idx * class_count + class_id];
            if table_prefetch {
                if let Some(next_byte) = next_byte {
                    let next_class = classes[usize::from(next_byte)] as usize;
                    let next_state_idx = (state & MASK_STATE) as usize;
                    prefetch_transition(table, next_state_idx * class_count + next_class);
                }
            }

            if (state & 0x4000_0000) != 0 {
                state = dfa.start_state();
                let restart_class = classes[usize::from(byte)] as usize;
                let restart_state_idx = (state & MASK_STATE) as usize;
                state = table[restart_state_idx * class_count + restart_class];
                if table_prefetch {
                    if let Some(next_byte) = next_byte {
                        let next_class = classes[usize::from(next_byte)] as usize;
                        let next_state_idx = (state & MASK_STATE) as usize;
                        prefetch_transition(table, next_state_idx * class_count + next_class);
                    }
                }
                if (state & 0x4000_0000) != 0 {
                    state = dfa.start_state();
                }
            }
        }

        black_box(state)
    }

    let patterns = build_large_regex_dfa();
    let dfa = &patterns.ir().regex_dfas()[0];
    let data = vec![b'z'; FILE_SIZE];
    let state_count = dfa.transition_table().len() / dfa.class_count() as usize;

    let mut group = c.benchmark_group("cpu_scan/regex_dfa_prefetch_100mb");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(FILE_SIZE as u64));

    group.bench_function(
        BenchmarkId::new("baseline", format!("{state_count}_states")),
        |b| {
            b.iter(|| {
                black_box(walk_dfa(false, dfa, &data));
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("prefetch", format!("{state_count}_states")),
        |b| {
            b.iter(|| {
                black_box(walk_dfa(true, dfa, &data));
            });
        },
    );

    group.finish();
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{}MB", bytes / 1_048_576)
    } else if bytes >= 1_024 {
        format!("{}KB", bytes / 1_024)
    } else {
        format!("{}B", bytes)
    }
}

criterion_group!(
    benches,
    bench_input_sizes,
    bench_pattern_counts,
    bench_security_scan,
    bench_realistic_patterns,
    bench_pattern_scale_100mb,
    bench_regex_dfa_prefetch_100mb,
);
criterion_main!(benches);
