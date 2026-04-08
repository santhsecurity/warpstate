use warpstate::*;

#[test]
fn test_null_bytes_in_patterns() {
    let ps = PatternSet::builder()
        .literal_bytes(b"A\x00B\x00C")
        .build()
        .unwrap();

    let data = b"prefixA\x00B\x00Csuffix";
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 6);
    assert_eq!(matches[0].end, 11);
}

#[test]
fn test_pattern_longer_than_input() {
    let ps = PatternSet::builder()
        .literal("THIS_PATTERN_IS_LONGER_THAN_THE_INPUT")
        .build()
        .unwrap();

    let data = b"SHORT_INPUT";
    let matches = ps.scan(data).unwrap();

    // Must not match and must not panic.
    assert_eq!(matches.len(), 0);
}
