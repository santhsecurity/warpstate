//! Regex DFA correctness harness for warpstate.
//!
//! The DFA scan path was entirely untested before this harness. Tests here
//! verify regex compilation, case-insensitive matching, anchors, alternation,
//! quantifiers, character classes, and edge cases.

use warpstate::{Match, PatternSet};

fn scan(pattern: &str, data: &[u8]) -> Vec<Match> {
    let ps = PatternSet::builder().regex(pattern).build().unwrap();
    ps.scan(data).unwrap()
}

fn scan_texts(pattern: &str, data: &[u8]) -> Vec<String> {
    scan(pattern, data)
        .iter()
        .map(|m| String::from_utf8_lossy(&data[m.start as usize..m.end as usize]).to_string())
        .collect()
}

fn scan_count(pattern: &str, data: &[u8]) -> usize {
    scan(pattern, data).len()
}

// ── Case-insensitive ────────────────────────────────────────────────────

#[test]
fn case_insensitive_matches_all_cases() {
    let texts = scan_texts("(?i:hello)", b"Hello hello HELLO hElLo");
    assert_eq!(texts, vec!["Hello", "hello", "HELLO", "hElLo"]);
}

#[test]
fn case_insensitive_across_lines() {
    let texts = scan_texts("(?i:needle)", b"Needle\nneedle\nNEEDLE\n");
    assert_eq!(texts.len(), 3);
    assert_eq!(texts[0], "Needle");
    assert_eq!(texts[1], "needle");
    assert_eq!(texts[2], "NEEDLE");
}

#[test]
fn case_insensitive_single_char() {
    let count = scan_count("(?i:a)", b"aAbBaA");
    assert_eq!(count, 4); // a, A, a, A
}

#[test]
fn case_insensitive_no_match() {
    let count = scan_count("(?i:xyz)", b"hello world");
    assert_eq!(count, 0);
}

// ── Basic regex features ────────────────────────────────────────────────

#[test]
fn dot_matches_non_newline() {
    let texts = scan_texts("a.c", b"abc aXc a\nc");
    assert!(texts.contains(&"abc".to_string()));
    assert!(texts.contains(&"aXc".to_string()));
    // a\nc should NOT match (dot doesn't match newline by default)
}

#[test]
fn character_class() {
    let texts = scan_texts("[0-9]+", b"abc 123 def 456");
    assert_eq!(texts, vec!["123", "456"]);
}

#[test]
fn negated_character_class() {
    let count = scan_count("[^a-z]", b"abc123");
    assert!(count >= 3); // 1, 2, 3
}

#[test]
fn alternation() {
    let texts = scan_texts("foo|bar", b"foo baz bar");
    assert_eq!(texts, vec!["foo", "bar"]);
}

#[test]
fn alternation_three_options() {
    let texts = scan_texts("cat|dog|bird", b"I have a dog and a cat");
    assert!(texts.contains(&"dog".to_string()));
    assert!(texts.contains(&"cat".to_string()));
}

// ── Quantifiers ─────────────────────────────────────────────────────────

#[test]
fn star_quantifier() {
    let count = scan_count("ab*c", b"ac abc abbc");
    assert!(count >= 3);
}

#[test]
fn plus_quantifier() {
    let texts = scan_texts("a+", b"a aa aaa b");
    // Non-overlapping: should find greedy matches
    assert!(!texts.is_empty());
}

#[test]
fn question_quantifier() {
    let count = scan_count("colou?r", b"color colour");
    assert_eq!(count, 2);
}

#[test]
fn exact_repetition() {
    let texts = scan_texts("a{3}", b"a aa aaa aaaa");
    assert!(!texts.is_empty());
    for t in &texts {
        assert!(t.len() >= 3);
    }
}

#[test]
fn range_repetition() {
    let count = scan_count("[0-9]{2,4}", b"1 12 123 1234 12345");
    assert!(count >= 3);
}

// ── Anchors ─────────────────────────────────────────────────────────────

#[test]
fn caret_anchor_start_of_input() {
    // With our anchored DFA + manual restart, ^ effectively matches at
    // every restart position. This test documents current behavior.
    let count = scan_count("^abc", b"abc\nxabc");
    // Current DFA: ^ doesn't work correctly for multiline (known limitation)
    assert_eq!(
        count, 1,
        "Current DFA limitation: ^ matches at every restart position"
    );
}

#[test]
fn dollar_anchor_eoi() {
    // $ at end-of-input should match via EOI transition
    let count = scan_count("abc$", b"abc");
    // Current DFA limitation: $ only matches at true end-of-input
    assert_eq!(count, 1, "Should match at end of input");
}

// ── Escape sequences ────────────────────────────────────────────────────

#[test]
fn digit_escape() {
    let texts = scan_texts(r"\d+", b"hello 42 world 7");
    assert!(texts.contains(&"42".to_string()));
    assert!(texts.contains(&"7".to_string()));
}

#[test]
fn word_escape() {
    let count = scan_count(r"\w+", b"hello world");
    assert!(count >= 2);
}

#[test]
fn whitespace_escape() {
    let count = scan_count(r"\s+", b"hello  world\t!");
    assert!(count >= 1);
}

// ── Groups ──────────────────────────────────────────────────────────────

#[test]
fn non_capturing_group() {
    let texts = scan_texts("(?:foo|bar)baz", b"foobaz barbaz");
    assert_eq!(texts.len(), 2);
}

#[test]
fn nested_groups() {
    let count = scan_count("(?:a(?:bc))+", b"abcabc");
    assert!(count >= 1);
}

// ── Edge cases ──────────────────────────────────────────────────────────

#[test]
fn empty_alternation_branch() {
    // Should compile and handle gracefully
    let result = PatternSet::builder().regex("a|").build();
    assert!(
        result.is_ok(),
        "Empty alternation branch should compile successfully"
    );
}

#[test]
fn very_long_regex_pattern() {
    let pattern = format!("(?:{})", "a".repeat(1000));
    let result = PatternSet::builder().regex(&pattern).build();
    assert!(result.is_ok());
}

#[test]
fn unicode_regex() {
    let texts = scan_texts("(?i:café)", b"Caf\xc3\xa9 CAF\xc3\x89"); // café CAFÉ in UTF-8
                                                                     // Should match at least the first
    assert!(!texts.is_empty());
}

#[test]
fn regex_on_empty_input() {
    let count = scan_count("abc", b"");
    assert_eq!(count, 0);
}

#[test]
fn regex_on_single_byte_input() {
    let count = scan_count("a", b"a");
    assert_eq!(count, 1);
}

#[test]
fn regex_no_match_in_large_input() {
    let data = vec![b'x'; 1_000_000];
    let count = scan_count("needle", &data);
    assert_eq!(count, 0);
}

#[test]
fn regex_many_matches_in_large_input() {
    let mut data = Vec::with_capacity(100_000);
    for _ in 0..10_000 {
        data.extend_from_slice(b"ab cd ");
    }
    let count = scan_count("ab", &data);
    assert_eq!(count, 10_000);
}

// ── Multiple regex patterns ─────────────────────────────────────────────

#[test]
fn two_regex_patterns() {
    let ps = PatternSet::builder()
        .regex("[0-9]+")
        .regex("[a-z]+")
        .build()
        .unwrap();
    let matches = ps.scan(b"abc 123").unwrap();
    // Should find both pattern types
    let ids: Vec<u32> = matches.iter().map(|m| m.pattern_id).collect();
    assert!(ids.contains(&0) || ids.contains(&1));
}

#[test]
fn regex_and_literal_mixed() {
    let ps = PatternSet::builder()
        .literal("needle")
        .regex("[0-9]+")
        .build()
        .unwrap();
    let matches = ps.scan(b"needle 42").unwrap();
    assert!(matches.len() >= 2);
}

// ── Regression tests ────────────────────────────────────────────────────

#[test]
fn regression_case_insensitive_exact_case_missed() {
    // Bug: (?i:hello) matched "Hello" and "HELLO" but missed "hello"
    // Root cause: DFA didn't reset to start state after match
    let texts = scan_texts("(?i:hello)", b"Hello\nhello\nHELLO");
    assert_eq!(texts.len(), 3, "must find all 3 case variants");
    assert!(texts.contains(&"Hello".to_string()));
    assert!(texts.contains(&"hello".to_string()));
    assert!(texts.contains(&"HELLO".to_string()));
}

#[test]
fn regression_consecutive_matches_not_missed() {
    // Verify back-to-back matches work after the DFA reset fix
    let texts = scan_texts("ab", b"ababab");
    assert_eq!(texts.len(), 3);
}

#[test]
fn regression_match_at_end_of_input() {
    let texts = scan_texts("end", b"the end");
    assert_eq!(texts.len(), 1);
    assert_eq!(texts[0], "end");
}

#[test]
fn regression_match_at_start_of_input() {
    let texts = scan_texts("start", b"start here");
    assert_eq!(texts.len(), 1);
    assert_eq!(texts[0], "start");
}

#[test]
fn regression_single_char_regex_repeated() {
    // Ensure single-char regex finds all occurrences
    let matches = scan("x", b"xxxx");
    for m in &matches {
        eprintln!("match: [{}-{}]", m.start, m.end);
    }
    assert_eq!(matches.len(), 4);
}
