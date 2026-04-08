use warpstate::*;

#[test]
fn test_unicode_combining_characters() {
    // e + acute accent (U+0301)
    let ps = PatternSet::builder().literal("e\u{0301}").build().unwrap();

    let data = "cafe\u{0301}s".as_bytes();
    let matches = ps.scan(data).unwrap();

    assert_eq!(matches.len(), 1);
    // "caf" is 3 bytes, "e" is 1 byte, U+0301 is 2 bytes in UTF-8
    assert_eq!(matches[0].start, 3);
    assert_eq!(matches[0].end, 6);
}
