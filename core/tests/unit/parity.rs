#![cfg(feature = "gpu")]
use warpstate::GpuMatcher;
use warpstate::{Error, PatternSet};

#[test]
fn standard_gpu_backend_scans_literal_input() {
    let patterns = PatternSet::builder()
        .literal("needle")
        .build()
        .expect("pattern set");
    let matcher = match pollster::block_on(GpuMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected standard GPU init error: {other:?}"),
    };

    let input = b"xxneedlezz";
    let gpu_matches = pollster::block_on(matcher.scan(input)).expect("standard scan");
    assert_eq!(gpu_matches, patterns.scan(input).expect("cpu scan"));
}

#[test]
fn consolidated_gpu_backend_scans_regex_input() {
    let patterns = PatternSet::builder()
        .regex("ab+c")
        .build()
        .expect("pattern set");
    let matcher = match pollster::block_on(GpuMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected consolidated GPU init error: {other:?}"),
    };

    let input = b"xxabbbcxx";
    let gpu_matches = pollster::block_on(matcher.scan(input)).expect("consolidated scan");
    assert!(!gpu_matches.is_empty());
    assert_eq!(gpu_matches[0].start, 2);
}
