use warpstate::algebraic::AlgebraicDfaMatcher;
use warpstate::gpu::GpuMatcher;
use warpstate::gpu_smem::SmemDfaMatcher;
use warpstate::matcher::BlockMatcher;
use warpstate::persistent::PersistentMatcher;
use warpstate::pipeline::StreamPipeline;
use warpstate::{Error, Matcher, PatternSet};

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
fn persistent_gpu_backend_scans_regex_input() {
    let patterns = PatternSet::builder()
        .regex("ab+c")
        .build()
        .expect("pattern set");
    let matcher = match pollster::block_on(PersistentMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected persistent init error: {other:?}"),
    };

    let input = b"xxabbbcxx";
    let gpu_matches = pollster::block_on(matcher.scan_block(input)).expect("persistent scan");
    assert!(!gpu_matches.is_empty());
    assert_eq!(gpu_matches[0].start, 2);
}

#[test]
fn persistent_zero_copy_path_matches_standard_path() {
    let patterns = PatternSet::builder()
        .regex("ab+c")
        .regex("needle")
        .build()
        .expect("pattern set");
    let matcher = match pollster::block_on(PersistentMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected persistent init error: {other:?}"),
    };
    let pipeline = StreamPipeline::new(matcher, 16_384);
    let input = b"zzabbbcxxneedlezz";

    let standard = pollster::block_on(pipeline.scan(input)).expect("standard scan");
    let mut zero_copy = pipeline
        .scan_zero_copy(input.len())
        .expect("zero-copy session");
    zero_copy.mapped_mut().copy_from_slice(input);
    let staged = pollster::block_on(zero_copy.finish()).expect("zero-copy scan");

    assert_eq!(staged, standard);
}

#[test]
fn persistent_zero_copy_path_handles_empty_input() {
    let patterns = PatternSet::builder()
        .regex("needle")
        .build()
        .expect("pattern set");
    let matcher = match pollster::block_on(PersistentMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected persistent init error: {other:?}"),
    };
    let pipeline = StreamPipeline::new(matcher, 16_384);

    let zero_copy = pipeline.scan_zero_copy(0).expect("zero-copy session");
    let staged = pollster::block_on(zero_copy.finish()).expect("zero-copy scan");

    assert!(staged.is_empty());
}

#[test]
fn smem_gpu_backend_matches_cpu() {
    let patterns = PatternSet::builder()
        .regex("needle|needlf")
        .build()
        .expect("pattern set");
    let matcher = match pollster::block_on(SmemDfaMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected smem init error: {other:?}"),
    };

    let input = b"xxneedleyy";
    let gpu_matches = pollster::block_on(matcher.scan_block(input)).expect("smem scan");
    assert_eq!(gpu_matches, patterns.scan(input).expect("cpu scan"));
}

#[test]
fn algebraic_gpu_backend_handles_non_matching_input() {
    let patterns = PatternSet::builder()
        .regex("needle|needlf")
        .build()
        .expect("pattern set");
    let matcher = match pollster::block_on(AlgebraicDfaMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected algebraic init error: {other:?}"),
    };

    let input = b"xxnomatchyy";
    let gpu_matches = pollster::block_on(matcher.scan_block(input)).expect("algebraic scan");
    assert_eq!(gpu_matches, patterns.scan(input).expect("cpu scan"));
}
