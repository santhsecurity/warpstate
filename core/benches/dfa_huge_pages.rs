//! Benchmark DFA transition-table throughput on a fixed 50MB table.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const TRANSITION_TABLE_BYTES: usize = 50 * 1_048_576;
const INPUT_BYTES: usize = 64 * 1_048_576;
const CLASS_COUNT: usize = 256;
const ENTRY_COUNT: usize = TRANSITION_TABLE_BYTES / std::mem::size_of::<u32>();
const STATE_COUNT: usize = ENTRY_COUNT / CLASS_COUNT;

enum TransitionTableBuffer {
    Standard(Vec<u32>),
    #[cfg(target_os = "linux")]
    HugePages {
        ptr: std::ptr::NonNull<u32>,
        len: usize,
        byte_len: usize,
    },
}

impl TransitionTableBuffer {
    fn standard(count: usize) -> Self {
        let mut values = vec![0u32; count];
        fill_transition_table(&mut values);
        Self::Standard(values)
    }

    #[cfg(target_os = "linux")]
    fn huge_pages(count: usize) -> Self {
        let byte_len = count
            .checked_mul(std::mem::size_of::<u32>())
            .expect("transition table byte length overflowed");
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Self::standard(count);
        }

        let ptr = std::ptr::NonNull::new(ptr.cast::<u32>()).expect("mmap returned a null pointer");
        let values = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), count) };
        fill_transition_table(values);
        Self::HugePages {
            ptr,
            len: count,
            byte_len,
        }
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Standard(values) => values.as_slice(),
            #[cfg(target_os = "linux")]
            Self::HugePages { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for TransitionTableBuffer {
    fn drop(&mut self) {
        if let Self::HugePages { ptr, byte_len, .. } = self {
            unsafe {
                let _ = libc::munmap(ptr.as_ptr().cast::<libc::c_void>(), *byte_len);
            }
        }
    }
}

fn fill_transition_table(values: &mut [u32]) {
    for state in 0..STATE_COUNT {
        let row = &mut values[state * CLASS_COUNT..(state + 1) * CLASS_COUNT];
        for (class_id, slot) in row.iter_mut().enumerate() {
            let next = ((state as u32).wrapping_mul(33) ^ class_id as u32) % STATE_COUNT as u32;
            *slot = next;
        }
    }
}

fn walk_transition_table(table: &[u32], haystack: &[u8]) -> u32 {
    let mut state = 0usize;
    for &byte in haystack {
        state = table[state * CLASS_COUNT + usize::from(byte)] as usize;
    }
    state as u32
}

fn bench_dfa_huge_pages(c: &mut Criterion) {
    let data: Vec<u8> = (0..INPUT_BYTES).map(|index| (index & 0xFF) as u8).collect();
    let standard = TransitionTableBuffer::standard(ENTRY_COUNT);

    let mut group = c.benchmark_group("dfa_transition_table_50mb");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("standard_pages", ENTRY_COUNT),
        &standard,
        |b, table| {
            b.iter(|| black_box(walk_transition_table(table.as_slice(), &data)));
        },
    );

    #[cfg(target_os = "linux")]
    {
        let huge_pages = TransitionTableBuffer::huge_pages(ENTRY_COUNT);
        group.bench_with_input(
            BenchmarkId::new("huge_pages", ENTRY_COUNT),
            &huge_pages,
            |b, table| {
                b.iter(|| black_box(walk_transition_table(table.as_slice(), &data)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dfa_huge_pages);
criterion_main!(benches);
