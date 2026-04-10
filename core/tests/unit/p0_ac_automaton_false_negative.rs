//! P0 REGRESSION TEST: AC Automaton false negative at 50+ patterns
//!
//! CRITICAL SECURITY BUG: The Aho-Corasick automaton in warpstate was not finding
//! patterns when PatternSet contained 50+ patterns, specifically affecting the
//! 'burpcollaborator%2Enet' pattern detection.
//!
//! ROOT CAUSE:
//! - AC automaton ID mapping issues at pattern count boundaries
//! - Pattern compilation succeeded but scan failed to match
//! - Silent failure - no error, just missing matches
//!
//! FIX VERIFICATION:
//! This test verifies that PatternSet with 50+ patterns correctly finds ALL
//! patterns, including 'burpcollaborator%2Enet' at various positions.
//! IF THIS TEST FAILS, THE ENGINE HAS A FALSE NEGATIVE - DO NOT WEAKEN.

use warpstate::PatternSet;

/// P0 REGRESSION: PatternSet with 50+ patterns must find 'burpcollaborator%2Enet'
///
/// This was the exact pattern that triggered the P0 bug. When PatternSet had
/// 37+ patterns, this pattern was silently not found despite being compiled.
/// The bug was in the AC automaton ID mapping at scale.
#[test]
fn p0_ac_automaton_finds_burpcollaborator_at_50_patterns() {
    let target = "burpcollaborator%2Enet";
    let test_data = format!(
        "GET /callback?domain={}&port=443 HTTP/1.1\r\nHost: evil.com\r\n\r\n",
        target
    );

    // Build PatternSet with 50+ patterns
    let mut builder = PatternSet::builder();

    // Add target pattern first
    builder = builder.literal(target);

    // Add 49 filler patterns to reach 50 total
    for i in 0..49 {
        builder = builder.literal(&format!("filler_pattern_{:04}.example.com/path", i));
    }

    let pattern_set = builder.build().expect(
        "CRITICAL: Failed to build PatternSet with 50 patterns. \
         Pattern compilation should succeed.",
    );

    assert_eq!(
        pattern_set.len(),
        50,
        "CRITICAL: PatternSet should have exactly 50 patterns, got {}",
        pattern_set.len()
    );

    // Scan for the target pattern
    let matches = pattern_set
        .scan(test_data.as_bytes())
        .expect("CRITICAL: Pattern scan failed");

    // Verify target pattern was found
    let found = matches
        .iter()
        .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == target.as_bytes());

    // CRITICAL ASSERTION - DO NOT WEAKEN
    // If this fails, the AC automaton is dropping patterns at scale
    assert!(
        found,
        "CRITICAL P0 REGRESSION: Pattern '{}' NOT FOUND with 50 patterns!\n\
         This is the exact false negative that caused the P0 incident.\n\
         The AC automaton is not finding patterns at 50+ pattern scale.\n\
         Matches found: {:?}\n\
         Test data: {}",
        target, matches, test_data
    );
}

/// P0 REGRESSION: PatternSet with 50+ patterns must find ALL patterns, not just one
///
/// This test verifies that every single pattern in a 50-pattern set is found
/// when scanning data containing all patterns. This catches ID mapping bugs
/// that might only affect some patterns.
#[test]
fn p0_ac_automaton_finds_all_patterns_at_50() {
    // Create 50 unique patterns
    let patterns: Vec<String> = (0..50)
        .map(|i| format!("unique.test.pattern.{:04}.marker.net", i))
        .collect();

    let mut builder = PatternSet::builder();
    for p in &patterns {
        builder = builder.literal(p);
    }

    let pattern_set = builder.build().expect("Failed to build PatternSet");
    assert_eq!(pattern_set.len(), 50);

    // Create test data containing ALL patterns
    let mut test_data = String::with_capacity(5000);
    test_data.push_str("START:");
    for p in &patterns {
        test_data.push_str("[PATTERN:");
        test_data.push_str(p);
        test_data.push_str("]");
    }
    test_data.push_str(":END");

    let matches = pattern_set.scan(test_data.as_bytes()).expect("Scan failed");

    // Verify EACH pattern was found
    let mut missing_patterns = Vec::new();
    for (idx, pattern) in patterns.iter().enumerate() {
        let found = matches
            .iter()
            .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == pattern.as_bytes());
        if !found {
            missing_patterns.push((idx, pattern.clone()));
        }
    }

    // CRITICAL ASSERTION
    assert!(
        missing_patterns.is_empty(),
        "CRITICAL P0 REGRESSION: {} patterns NOT FOUND at 50-pattern scale!\n\
         Missing patterns: {:?}\n\
         This indicates the AC automaton is dropping patterns.\n\
         Total matches found: {} (expected 50)",
        missing_patterns.len(),
        missing_patterns,
        matches.len()
    );
}

/// P0 REGRESSION: Pattern position in set must not affect findability
///
/// The original bug manifested when the target pattern was at different positions
/// in the PatternSet. This test verifies the pattern is found regardless of
/// its position in the set.
#[test]
fn p0_ac_automaton_pattern_position_independence() {
    let target = "burpcollaborator%2Enet";
    let test_data = format!("xxx {} yyy", target);

    // Test pattern at various positions
    for position in [0, 10, 25, 37, 49] {
        let mut builder = PatternSet::builder();

        // Add patterns before target
        for i in 0..position {
            builder = builder.literal(&format!("before_{:04}", i));
        }

        // Add target at position
        builder = builder.literal(target);

        // Add patterns after target to reach 50 total
        for i in position..49 {
            builder = builder.literal(&format!("after_{:04}", i));
        }

        let pattern_set = builder.build().expect("Failed to build PatternSet");
        let matches = pattern_set.scan(test_data.as_bytes()).expect("Scan failed");

        let found = matches
            .iter()
            .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == target.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Pattern NOT FOUND at position {} in 50-pattern set!\n\
             Position in PatternSet should not affect findability.\n\
             Matches: {:?}",
            position, matches
        );
    }
}

/// P0 REGRESSION: PatternSet boundaries must not cause drops
///
/// Tests various pattern count boundaries where strategy transitions occur:
/// - 8 patterns: MultiMemchr -> AhoCorasick transition
/// - 37 patterns: Original bug boundary
/// - 50 patterns: HashScanner threshold
/// - 100 patterns: Large set handling
#[test]
fn p0_ac_automaton_boundary_pattern_counts() {
    let target = "burpcollaborator%2Enet";
    let test_data = format!("GET /?c2={} HTTP/1.1", target);

    for count in [8, 9, 10, 37, 50, 51, 100] {
        let mut builder = PatternSet::builder();
        builder = builder.literal(target);

        // Add filler patterns to reach target count
        for i in 0..(count - 1) {
            builder = builder.literal(&format!("boundary_filler_{:04}", i));
        }

        let pattern_set = builder.build().expect("Failed to build PatternSet");
        assert_eq!(
            pattern_set.len(),
            count,
            "PatternSet should have {} patterns",
            count
        );

        let matches = pattern_set.scan(test_data.as_bytes()).expect("Scan failed");

        let found = matches
            .iter()
            .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == target.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Pattern NOT FOUND at {} patterns!\n\
             Pattern count boundaries must not cause pattern drops.\n\
             This is a security vulnerability.",
            count
        );
    }
}

/// P0 REGRESSION: Percent-encoded patterns must be found at scale
///
/// The original bug specifically affected 'burpcollaborator%2Enet' which contains
/// percent-encoding. This test verifies various percent-encoded patterns work.
#[test]
fn p0_ac_automaton_percent_encoded_patterns() {
    let percent_patterns = vec![
        "burpcollaborator%2Enet",
        "example%2Ecom%2Fpath",
        "test%2Eorg%2Fapi%2Fv1",
        "%2Fadmin%2Flogin%2Ephp",
        "callback%3Fid%3D123",
    ];

    for target in &percent_patterns {
        let mut builder = PatternSet::builder();
        builder = builder.literal(*target);

        // Add 49 filler patterns
        for i in 0..49 {
            builder = builder.literal(&format!("filler{:04}.test.com", i));
        }

        let pattern_set = builder.build().expect("Failed to build PatternSet");
        let test_data = format!("prefix {} suffix", target);
        let matches = pattern_set.scan(test_data.as_bytes()).expect("Scan failed");

        let found = matches
            .iter()
            .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == target.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Percent-encoded pattern '{}' NOT FOUND!\n\
             Percent-encoded patterns are critical for C2 detection.\n\
             Matches: {:?}",
            target, matches
        );
    }
}

/// P0 REGRESSION: Large pattern sets must not drop patterns
///
/// Tests pattern sets up to 5000 patterns to ensure the HashScanner boundary
/// and large-set handling don't cause pattern drops.
#[test]
fn p0_ac_automaton_large_set_no_drops() {
    let target = "burpcollaborator%2Enet";
    let test_data = format!("GET /?x={}", target);

    for count in [100, 500, 1000, 4999, 5000, 5001] {
        let mut builder = PatternSet::builder();
        builder = builder.literal(target);

        // Add filler patterns
        for i in 0..(count - 1) {
            builder = builder.literal(&format!("large_set_filler_{:08}", i));
        }

        let pattern_set = builder.build().expect("Failed to build PatternSet");
        assert_eq!(pattern_set.len(), count);

        let matches = pattern_set.scan(test_data.as_bytes()).expect("Scan failed");

        let found = matches
            .iter()
            .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == target.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Pattern NOT FOUND with {} patterns!\n\
             Large pattern sets must not drop patterns.\n\
             This is the HashScanner boundary or large-set handling bug.",
            count
        );
    }
}

/// P0 REGRESSION: Multiple matches of same pattern must all be found
///
/// Ensures that when a pattern appears multiple times in data, all instances
/// are found, not just the first.
#[test]
fn p0_ac_automaton_multiple_matches_all_found() {
    let target = "burpcollaborator%2Enet";

    let mut builder = PatternSet::builder();
    builder = builder.literal(target);

    // Add 49 fillers
    for i in 0..49 {
        builder = builder.literal(&format!("multi_filler_{}", i));
    }

    let pattern_set = builder.build().expect("Failed to build PatternSet");

    // Create data with pattern appearing 5 times
    let test_data = format!("{} {} {} {} {}", target, target, target, target, target);

    let matches = pattern_set.scan(test_data.as_bytes()).expect("Scan failed");

    // Count matches of target pattern
    let target_matches: Vec<_> = matches
        .iter()
        .filter(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == target.as_bytes())
        .collect();

    assert_eq!(
        target_matches.len(),
        5,
        "CRITICAL P0 REGRESSION: Expected 5 matches of pattern, found {}!\n\
         The AC automaton is not finding all pattern instances.\n\
         All matches: {:?}",
        target_matches.len(),
        matches
    );
}
