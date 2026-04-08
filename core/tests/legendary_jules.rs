use std::sync::Arc;
use warpstate::{Match, Matcher, PatternSet};

#[cfg(feature = "gpu")]
use warpstate::gpu::GpuMatcher;

// Helper to check match positions exactly
#[track_caller]
fn assert_matches(matches: &[Match], expected: &[(u32, u32)]) {
    let actual_pos: Vec<(u32, u32)> = matches.iter().map(|m| (m.start, m.end)).collect();
    assert_eq!(actual_pos, expected);
}

// 1. BOUNDARY: empty input, 1 byte, pattern at byte 0, pattern at last byte, pattern spanning end
#[test]
fn test_boundary_conditions() {
    let ps = PatternSet::builder()
        .literal("a")
        .literal("boundary")
        .build()
        .expect("build pattern set");

    // Empty input
    let matches = ps.scan(b"").unwrap();
    assert_matches(&matches, &[]);

    // 1 byte input matching
    let matches = ps.scan(b"a").unwrap();
    assert_matches(&matches, &[(0, 1)]);

    // 1 byte input non-matching
    let matches = ps.scan(b"b").unwrap();
    assert_matches(&matches, &[]);

    // Pattern at byte 0
    let matches = ps.scan(b"boundary_test").unwrap();
    assert_matches(&matches, &[(0, 8)]);

    // Pattern at last byte
    let matches = ps.scan(b"test_boundary").unwrap();
    assert_matches(&matches, &[(5, 13)]);

    // Pattern spanning end (partially matching at the end shouldn't match)
    let matches = ps.scan(b"test_boundar").unwrap();
    assert_matches(&matches, &[(10, 11)]);
}
