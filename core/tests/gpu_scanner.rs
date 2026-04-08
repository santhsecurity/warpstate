use std::sync::Arc;

use warpstate::{GpuScanner, PatternSet};

#[test]
fn gpu_scanner_matches_pattern_set_for_mixed_patterns() {
    let patterns = PatternSet::builder()
        .literal("eval(")
        .regex(r"child_[a-z]+")
        .build()
        .unwrap();
    let scanner = GpuScanner::new(patterns.clone());
    let input = b"eval('x'); require('child_process');";

    let expected = patterns.scan(input).unwrap();
    let actual = scanner.scan(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn gpu_scanner_is_shareable_across_threads() {
    let patterns = PatternSet::builder()
        .literal("needle")
        .regex(r"thread_[0-9]+")
        .build()
        .unwrap();
    let scanner = Arc::new(GpuScanner::new(patterns.clone()));
    let expected = patterns.scan(b"needle thread_42 needle").unwrap();

    let mut threads = Vec::new();
    for _ in 0..8 {
        let scanner = Arc::clone(&scanner);
        let expected = expected.clone();
        threads.push(std::thread::spawn(move || {
            for _ in 0..32 {
                let matches = scanner.scan(b"needle thread_42 needle").unwrap();
                assert_eq!(matches, expected);
            }
        }));
    }

    for thread in threads {
        thread.join().unwrap();
    }
}
