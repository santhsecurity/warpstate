use warpstate::*;

#[test]
fn single_pattern_fast_path_returns_all_matches() {
    let ps = PatternSet::builder().literal("foo").build().unwrap();
    let data = b"foo and foo and foo";

    let matches = ps.scan(data).unwrap();

    // Engine might have a fast path for single patterns.
    // Ensure it doesn't just stop at the first match.
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 8);
    assert_eq!(matches[2].start, 16);
}
