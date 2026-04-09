//! Exhaustive scan correctness harness.
//!
//! Generates random patterns and random data, then verifies that warpstate's
//! scan produces the SAME matches as a reference implementation (regex crate).
//! This is the D17 "SQLite TH3" approach — systematic exploration of the
//! combinatorial space rather than handcrafted test cases.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use warpstate::PatternSet;

/// Reference implementation using regex crate directly.
fn reference_scan(pattern: &str, data: &[u8]) -> Vec<(usize, usize)> {
    let Ok(re) = regex::bytes::Regex::new(pattern) else {
        return Vec::new();
    };
    re.find_iter(data).map(|m| (m.start(), m.end())).collect()
}

/// Generate a random literal pattern of given length.
/// Uses full printable ASCII (0x20-0x7E) not just [a-z] — Kimi correctly
/// flagged the narrow alphabet as a blind spot.
fn random_literal(rng: &mut StdRng, len: usize) -> String {
    (0..len)
        .map(|_| {
            // Printable ASCII excluding regex metacharacters for literal tests
            let c = loop {
                let b = rng.gen_range(0x20u8..=0x7E);
                if !b".*+?()[]{}|^$\\".contains(&b) {
                    break b;
                }
            };
            c as char
        })
        .collect()
}

/// Generate random data — full byte range for realistic testing.
fn random_data(rng: &mut StdRng, size: usize) -> Vec<u8> {
    (0..size).map(|_| rng.gen_range(0x20u8..=0x7E)).collect()
}

/// Generate random data with injected patterns at non-overlapping positions.
fn random_data_with_patterns(rng: &mut StdRng, size: usize, patterns: &[&str]) -> Vec<u8> {
    let mut data = random_data(rng, size);
    let mut used_ranges: Vec<(usize, usize)> = Vec::new();
    for pattern in patterns {
        let bytes = pattern.as_bytes();
        if bytes.len() >= size {
            continue;
        }
        // Try up to 10 times to find a non-overlapping position
        for _ in 0..10 {
            let pos = rng.gen_range(0..size - bytes.len());
            let end = pos + bytes.len();
            if !used_ranges.iter().any(|&(s, e)| pos < e && end > s) {
                data[pos..end].copy_from_slice(bytes);
                used_ranges.push((pos, end));
                break;
            }
        }
    }
    data
}

#[test]
fn exhaustive_single_literal_parity() {
    let mut failures = 0;

    // Multiple seeds to avoid deterministic blind spots (Kimi finding #6)
    for seed in [42, 99, 777, 12345, 0xDEAD] {
        let mut rng = StdRng::seed_from_u64(seed);

        for trial in 0..200 {
            let pat_len = rng.gen_range(1..=8);
            let pattern = random_literal(&mut rng, pat_len);
            let data_size = rng.gen_range(10..=10000);
            let data = random_data_with_patterns(&mut rng, data_size, &[&pattern]);

            let reference = reference_scan(&regex::escape(&pattern), &data);
            let ps = PatternSet::builder().literal(&pattern).build().unwrap();
            let warpstate: Vec<(usize, usize)> = ps
                .scan(&data)
                .unwrap()
                .iter()
                .map(|m| (m.start as usize, m.end as usize))
                .collect();

            if reference != warpstate {
                failures += 1;
                eprintln!(
                    "MISMATCH trial {trial}: pattern={pattern:?} data_len={data_size} ref={} ws={}",
                    reference.len(),
                    warpstate.len()
                );
            }
        }
    } // end seed loop
    assert_eq!(
        failures, 0,
        "{failures} parity failures in 1000 trials (5 seeds × 200)"
    );
}

#[test]
fn exhaustive_regex_parity() {
    let patterns = [
        r"[a-z]+",
        r"\d+",
        r"[aeiou]{2,4}",
        r"(abc|def|ghi)",
        r"[^aeiou]{3}",
        r"\w+\d",
    ];
    let mut rng = StdRng::seed_from_u64(99);
    let mut failures = 0;

    for (idx, pattern) in patterns.iter().enumerate() {
        for trial in 0..100 {
            let data_size = rng.gen_range(50..=5000);
            let data: Vec<u8> = (0..data_size)
                .map(|_| rng.gen_range(0x20..=0x7E) as u8)
                .collect();

            let reference = reference_scan(pattern, &data);
            let ps = match PatternSet::builder().regex(pattern).build() {
                Ok(ps) => ps,
                Err(_) => continue,
            };
            let warpstate: Vec<(usize, usize)> = ps
                .scan(&data)
                .unwrap()
                .iter()
                .map(|m| (m.start as usize, m.end as usize))
                .collect();

            if reference != warpstate {
                failures += 1;
                eprintln!(
                    "REGEX MISMATCH pattern[{idx}]={pattern:?} trial={trial} ref={} ws={}",
                    reference.len(),
                    warpstate.len()
                );
            }
        }
    }
    assert_eq!(failures, 0, "{failures} regex parity failures");
}

#[test]
fn exhaustive_empty_and_boundary() {
    // Empty input
    let ps = PatternSet::builder().literal("x").build().unwrap();
    assert_eq!(ps.scan(b"").unwrap().len(), 0);

    // Single byte input matching
    assert_eq!(ps.scan(b"x").unwrap().len(), 1);

    // Pattern at start
    assert_eq!(ps.scan(b"xaaa").unwrap()[0].start, 0);

    // Pattern at end
    let m = ps.scan(b"aaax").unwrap();
    assert_eq!(m[0].start, 3);
    assert_eq!(m[0].end, 4);

    // All bytes match
    let all_x = vec![b'x'; 1000];
    assert_eq!(ps.scan(&all_x).unwrap().len(), 1000);
}

#[test]
fn exhaustive_case_insensitive_parity() {
    let mut rng = StdRng::seed_from_u64(77);
    let mut failures = 0;

    for trial in 0..500 {
        let pat_len = rng.gen_range(1..=6);
        let pattern = random_literal(&mut rng, pat_len);
        let data_size = rng.gen_range(10..=5000);
        let mut data: Vec<u8> = (0..data_size).map(|_| rng.gen_range(b'a'..=b'z')).collect();
        // Randomly uppercase some bytes
        for byte in data.iter_mut() {
            if rng.gen_bool(0.3) {
                *byte = byte.to_ascii_uppercase();
            }
        }

        let ci_pattern = format!("(?i:{})", regex::escape(&pattern));
        let reference = reference_scan(&ci_pattern, &data);

        let ps = PatternSet::builder()
            .literal(&pattern)
            .case_insensitive(true)
            .build()
            .unwrap();
        let warpstate: Vec<(usize, usize)> = ps
            .scan(&data)
            .unwrap()
            .iter()
            .map(|m| (m.start as usize, m.end as usize))
            .collect();

        if reference != warpstate {
            failures += 1;
            eprintln!(
                "CI MISMATCH trial {trial}: pattern={pattern:?} ref={} ws={}",
                reference.len(),
                warpstate.len()
            );
        }
    }
    assert_eq!(failures, 0, "{failures} CI parity failures in 500 trials");
}
