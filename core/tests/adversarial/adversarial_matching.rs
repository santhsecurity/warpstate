use warpstate::PatternSet;

// TESTS 1-5: Overlapping literal patterns (aa, aaa, aaaa in 'aaaaa')

#[test]
fn test_01_overlap_aa_in_aaaaa() {
    let ps = PatternSet::builder().literal("aa").build().unwrap();
    let data = b"aaaaa";
    let matches = ps.scan_overlapping(data).unwrap();
    assert_eq!(matches.len(), 4);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 2);
    assert_eq!(matches[1].start, 1);
    assert_eq!(matches[1].end, 3);
    assert_eq!(matches[2].start, 2);
    assert_eq!(matches[2].end, 4);
    assert_eq!(matches[3].start, 3);
    assert_eq!(matches[3].end, 5);
}

#[test]
fn test_02_overlap_aaa_in_aaaaa() {
    let ps = PatternSet::builder().literal("aaa").build().unwrap();
    let data = b"aaaaa";
    let matches = ps.scan_overlapping(data).unwrap();
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 1);
    assert_eq!(matches[2].start, 2);
}

#[test]
fn test_03_overlap_aaaa_in_aaaaa() {
    let ps = PatternSet::builder().literal("aaaa").build().unwrap();
    let data = b"aaaaa";
    let matches = ps.scan_overlapping(data).unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 1);
}

#[test]
fn test_04_overlap_mixed_lengths() {
    let ps = PatternSet::builder()
        .literal("aa")
        .literal("aaa")
        .build()
        .unwrap();
    let data = b"aaaa";
    let matches = ps.scan_overlapping(data).unwrap();
    // "aa" -> 3 matches (0-2, 1-3, 2-4)
    // "aaa" -> 2 matches (0-3, 1-4)
    // Total 5 matches.
    assert_eq!(matches.len(), 5);
}

#[test]
fn test_05_overlap_identical_patterns() {
    // Adding the exact same pattern multiple times (or effectively same via regex vs literal)
    let ps = PatternSet::builder()
        .literal("aa")
        .regex("aa") // A simple regex that extracts into literal might be deduped, but regex scanning is done alongside literal scanning.
        .build()
        .unwrap();
    let data = b"aaa";
    // scan_overlapping includes both Aho-Corasick matches and RegexDFA native matches.
    // However, overlapping regex is fundamentally tricky. We just test what `scan_overlapping` produces.
    let matches = ps.scan_overlapping(data).unwrap();

    // literal 'aa' matches 0..2 and 1..3
    // regex 'aa' (which might be extracted to literal or run in DFA)
    assert!(matches.len() >= 2);
}

// TESTS 6-10: Regex patterns with alternation and repetition at input boundaries

#[test]
fn test_06_regex_alternation_boundary_start() {
    let ps = PatternSet::builder().regex("^(foo|bar)").build().unwrap();
    let data = b"foobaz";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 3);
}

#[test]
fn test_07_regex_alternation_boundary_end() {
    let ps = PatternSet::builder().regex("(foo|bar)$").build().unwrap();
    let data = b"bazbar";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 3);
    assert_eq!(matches[0].end, 6);
}

#[test]
fn test_08_regex_repetition_exact_boundary() {
    let ps = PatternSet::builder().regex("^a+$").build().unwrap();
    let data = b"aaaaa";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 5);

    let no_match = b"baaaaa";
    assert_eq!(ps.scan(no_match).unwrap().len(), 0);
}

#[test]
fn test_09_regex_repetition_leading() {
    let ps = PatternSet::builder().regex("^a+b").build().unwrap();
    let data = b"aaabx";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 4);
}

#[test]
fn test_10_regex_repetition_trailing() {
    let ps = PatternSet::builder().regex("ba+$").build().unwrap();
    let data = b"xbaaa";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 5);
}

// TESTS 11-15: Binary patterns with null bytes, 0xFF bytes, mixed

#[test]
fn test_11_binary_null_bytes() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0x00, 0x00, 0x00])
        .build()
        .unwrap();
    let data = vec![0x01, 0x00, 0x00, 0x00, 0x02];
    let matches = ps.scan(&data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 4);
}

#[test]
fn test_12_binary_ff_bytes() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0xFF, 0xFF])
        .build()
        .unwrap();
    let data = vec![0xFF, 0xFF, 0xFF];
    let matches = ps.scan_overlapping(&data).unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 1);
}

#[test]
fn test_13_binary_mixed_extremes() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0x00, 0xFF, 0x00, 0xFF])
        .build()
        .unwrap();
    let data = vec![0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF];
    let matches = ps.scan_overlapping(&data).unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 2);
}

#[test]
fn test_14_binary_with_regex() {
    // Note: Regex backend only partially supports arbitrary binary depending on encoding.
    // We stick to what compiles. \x00 compiles in regex crate if unicode is disabled, but here we test matching.
    let ps = PatternSet::builder()
        .regex(r"\x00\xFF\x00")
        .build()
        .unwrap();
    let data = vec![0x01, 0x00, 0xFF, 0x00, 0x02];
    let matches = ps.scan(&data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 4);
}

#[test]
fn test_15_binary_overlapping_nulls() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0x00, 0x00])
        .build()
        .unwrap();
    let data = vec![0x00; 5];
    let matches = ps.scan_overlapping(&data).unwrap();
    assert_eq!(matches.len(), 4);
}

// TESTS 16-20: Patterns longer than input (must not match, must not panic)

#[test]
fn test_16_pattern_longer_than_input_literal() {
    let ps = PatternSet::builder()
        .literal("longpattern")
        .build()
        .unwrap();
    let data = b"short";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_17_pattern_longer_than_input_regex() {
    let ps = PatternSet::builder().regex("a{10}").build().unwrap();
    let data = b"aaaaa";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_18_pattern_longer_than_input_binary() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0x01; 20])
        .build()
        .unwrap();
    let data = vec![0x01; 10];
    let matches = ps.scan(&data).unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_19_pattern_longer_than_input_mixed_set() {
    let ps = PatternSet::builder()
        .literal("longpattern1")
        .literal("longpattern2")
        .regex("longpattern3")
        .build()
        .unwrap();
    let data = b"short";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_20_pattern_longer_than_input_by_one_byte() {
    let ps = PatternSet::builder().literal("abcdef").build().unwrap();
    let data = b"abcde";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 0);
}

// TESTS 21-25: Empty input, single byte input, exactly-pattern-length input

#[test]
fn test_21_empty_input() {
    let ps = PatternSet::builder().literal("pattern").build().unwrap();
    let data = b"";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_22_single_byte_input() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let data = b"a";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 1);

    let no_match = b"b";
    assert_eq!(ps.scan(no_match).unwrap().len(), 0);
}

#[test]
fn test_23_exactly_pattern_length_input() {
    let ps = PatternSet::builder().literal("exact").build().unwrap();
    let data = b"exact";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 5);
}

#[test]
fn test_24_single_byte_regex() {
    let ps = PatternSet::builder().regex("^[a-z]$").build().unwrap();
    let data = b"z";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 1);
}

#[test]
fn test_25_exact_length_regex() {
    let ps = PatternSet::builder().regex("^12345$").build().unwrap();
    let data = b"12345";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 5);
}

// TESTS 26-30: Unicode patterns in UTF-8 encoded input

#[test]
fn test_26_unicode_literal_match() {
    let ps = PatternSet::builder().literal("🚀").build().unwrap();
    let data = "🚀".as_bytes();
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 4); // "🚀" is 4 bytes
}

#[test]
fn test_27_unicode_regex_match() {
    let ps = PatternSet::builder().regex("🔥+").build().unwrap();
    let data = "🔥🔥🔥".as_bytes();
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 12);
}

#[test]
fn test_28_unicode_mixed_with_ascii() {
    let ps = PatternSet::builder().literal("Hello 世界").build().unwrap();
    let data = "Hello 世界".as_bytes();
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 12); // "Hello " is 6 bytes + "世界" is 6 bytes
}

#[test]
fn test_29_unicode_boundary_overlap() {
    let ps = PatternSet::builder().literal("猫猫").build().unwrap();
    let data = "猫猫猫".as_bytes();
    let matches = ps.scan_overlapping(data).unwrap();
    assert_eq!(matches.len(), 2);
    // "猫" is 3 bytes (E7 8C AB)
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 3);
}

#[test]
fn test_30_unicode_invalid_utf8_input() {
    let ps = PatternSet::builder().literal("test").build().unwrap();
    // 0xFF is invalid UTF-8
    let data = vec![b't', b'e', b's', b't', 0xFF, b't', b'e', b's', b't'];
    let matches = ps.scan(&data).unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 5);
}

// TESTS 31-33: Maximum pattern count (1000+ patterns simultaneously)

#[test]
fn test_31_max_pattern_count_all_match() {
    let mut builder = PatternSet::builder();
    for i in 0..1005 {
        builder = builder.literal(&format!("PATTERN_{:04}", i));
    }
    let ps = builder.build().unwrap();

    // Create input that contains all 1005 patterns
    let mut data = String::new();
    for i in 0..1005 {
        data.push_str(&format!("PATTERN_{:04} ", i));
    }

    let matches = ps.scan(data.as_bytes()).unwrap();
    assert_eq!(matches.len(), 1005);
}

#[test]
fn test_32_max_pattern_count_none_match() {
    let mut builder = PatternSet::builder();
    for i in 0..1005 {
        builder = builder.literal(&format!("PATTERN_{:04}", i));
    }
    let ps = builder.build().unwrap();

    let data = b"completely unrelated text that does not match anything";
    let matches = ps.scan(data).unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_33_max_pattern_count_mixed_regex_literal() {
    let mut builder = PatternSet::builder();
    for i in 0..505 {
        builder = builder.literal(&format!("LIT_{:04}", i));
        builder = builder.regex(&format!("^REG_{:04}$", i));
    }
    let ps = builder.build().unwrap();

    // Total 1010 patterns. Match one literal and one regex.
    let data_lit = b"LIT_0250";
    let matches_lit = ps.scan(data_lit).unwrap();
    assert_eq!(matches_lit.len(), 1);
    assert_eq!(matches_lit[0].start, 0);
    assert_eq!(matches_lit[0].end, 8);

    let data_reg = b"REG_0400";
    let matches_reg = ps.scan(data_reg).unwrap();
    assert_eq!(matches_reg.len(), 1);
    assert_eq!(matches_reg[0].start, 0);
    assert_eq!(matches_reg[0].end, 8);
}
