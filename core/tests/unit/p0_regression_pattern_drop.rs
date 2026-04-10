//! P0 BUG REGRESSION TEST: PatternSet silently drops patterns at scale
//!
//! CRITICAL SECURITY BUG: warpscan scans the ENTIRE internet's software supply chain.
//! This bug means malware goes undetected.
//!
//! BUG DESCRIPTION:
//! - Pattern 'burpcollaborator%2Enet' is found when PatternSet has ~30 patterns
//! - Same pattern is NOT found when PatternSet has ~37+ patterns (adding abuse.ch URLhaus patterns)
//! - The pattern IS compiled into the PatternSet (compile reports correct count)
//! - But scan() does not find it in the input file
//!
//! ROOT CAUSE ANALYSIS:
//! The bug was potentially caused by:
//! 1. Hash collision in LiteralPrefilterTable (FNV-1a 32-bit hash has high collision probability)
//! 2. Aho-Corasick automaton ID mapping issues at scale
//! 3. ScanStrategy selection edge cases at pattern count boundaries
//!
//! FIX VERIFICATION:
//! These tests verify that all patterns are correctly found at various scales.

use warpstate::PatternSet;

/// Regression test: P0 bug with burpcollaborator%2Enet pattern at 30 patterns
#[test]
fn p0_burpcollaborator_found_at_30_patterns() {
    let target = "burpcollaborator%2Enet";
    let test_data = b"GET /callback?domain=burpcollaborator%2Enet HTTP/1.1";

    let mut builder = PatternSet::builder();
    builder = builder.literal(target);

    // Add 29 filler patterns
    for i in 0..29 {
        builder = builder.literal(&format!("malware{}.example.com/path", i));
    }

    let ps = builder.build().unwrap();
    assert_eq!(ps.len(), 30, "PatternSet should have 30 patterns");

    let matches = ps.scan(test_data).unwrap();
    let found = matches
        .iter()
        .any(|m| &test_data[m.start as usize..m.end as usize] == target.as_bytes());

    assert!(
        found,
        "CRITICAL P0 REGRESSION: Target pattern '{}' NOT FOUND with 30 patterns!\n\
         This is a security vulnerability - malware would go undetected.\n\
         Matches found: {:?}",
        target, matches
    );
}

/// Regression test: P0 bug with burpcollaborator%2Enet pattern at 37 patterns (exact bug boundary)
#[test]
fn p0_burpcollaborator_found_at_37_patterns() {
    let target = "burpcollaborator%2Enet";
    let test_data = b"GET /callback?domain=burpcollaborator%2Enet HTTP/1.1";

    let mut builder = PatternSet::builder();
    builder = builder.literal(target);

    // Add 36 filler patterns to reach 37 total
    for i in 0..36 {
        builder = builder.literal(&format!("malware{}.example.com/path", i));
    }

    let ps = builder.build().unwrap();
    assert_eq!(ps.len(), 37, "PatternSet should have 37 patterns");

    let matches = ps.scan(test_data).unwrap();
    let found = matches
        .iter()
        .any(|m| &test_data[m.start as usize..m.end as usize] == target.as_bytes());

    assert!(
        found,
        "CRITICAL P0 REGRESSION: Target pattern '{}' NOT FOUND with 37 patterns!\n\
         This was the exact boundary where the bug manifested.\n\
         Matches found: {:?}",
        target, matches
    );
}

/// Regression test: P0 bug with burpcollaborator%2Enet pattern at 50 patterns
#[test]
fn p0_burpcollaborator_found_at_50_patterns() {
    let target = "burpcollaborator%2Enet";
    let test_data = b"GET /callback?domain=burpcollaborator%2Enet HTTP/1.1";

    let mut builder = PatternSet::builder();
    builder = builder.literal(target);

    // Add 49 filler patterns
    for i in 0..49 {
        builder = builder.literal(&format!("malware{}.example.com/path", i));
    }

    let ps = builder.build().unwrap();
    assert_eq!(ps.len(), 50, "PatternSet should have 50 patterns");

    let matches = ps.scan(test_data).unwrap();
    let found = matches
        .iter()
        .any(|m| &test_data[m.start as usize..m.end as usize] == target.as_bytes());

    assert!(
        found,
        "CRITICAL P0 REGRESSION: Target pattern '{}' NOT FOUND with 50 patterns!\n\
         This is a security vulnerability - malware would go undetected.\n\
         Matches found: {:?}",
        target, matches
    );
}

/// Regression test: Verify ALL patterns are found at scale, not just the target
#[test]
fn p0_all_patterns_found_at_50_patterns() {
    let patterns: Vec<String> = (0..50)
        .map(|i| format!("unique.pattern.{:04}.test.com/path", i))
        .collect();

    let mut builder = PatternSet::builder();
    for p in &patterns {
        builder = builder.literal(p);
    }

    let ps = builder.build().unwrap();

    // Create test data containing ALL patterns
    let mut test_data = String::new();
    for p in &patterns {
        test_data.push_str("prefix ");
        test_data.push_str(p);
        test_data.push_str(" suffix ");
    }

    let matches = ps.scan(test_data.as_bytes()).unwrap();

    // Verify EACH pattern was found
    for (idx, pattern) in patterns.iter().enumerate() {
        let found = matches
            .iter()
            .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == pattern.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Pattern {} ('{}') was NOT FOUND! \
             This indicates patterns are being silently dropped at scale.",
            idx, pattern
        );
    }
}

/// Regression test: Percent-encoded patterns must be found at scale
#[test]
fn p0_percent_encoded_patterns_found_at_scale() {
    let percent_patterns = vec![
        "burpcollaborator%2Enet",
        "example%2Ecom",
        "test%2Eorg%2Fpath",
        "%2Fapi%2Fv1%2Fusers",
    ];

    for target in &percent_patterns {
        let mut builder = PatternSet::builder();
        builder = builder.literal(*target);

        // Add 49 filler patterns
        for i in 0..49 {
            builder = builder.literal(&format!("filler{:04}.example.com", i));
        }

        let ps = builder.build().unwrap();
        let test_data = format!("prefix {} suffix", target);
        let matches = ps.scan(test_data.as_bytes()).unwrap();

        let found = matches
            .iter()
            .any(|m| &test_data.as_bytes()[m.start as usize..m.end as usize] == target.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Percent-encoded pattern '{}' NOT FOUND!\n\
             Matches: {:?}",
            target, matches
        );
    }
}

/// Regression test: Pattern at various positions in the set must be found
#[test]
fn p0_pattern_at_all_positions_found() {
    let test_data = b"xxx TARGET_PATTERN yyy";

    for position in [0, 10, 20, 30, 35, 36, 49] {
        let mut builder = PatternSet::builder();

        // Add patterns before target
        for i in 0..position {
            builder = builder.literal(&format!("before{:04}", i));
        }

        // Add target at position
        builder = builder.literal("TARGET_PATTERN");

        // Add patterns after target
        for i in position..49 {
            builder = builder.literal(&format!("after{:04}", i));
        }

        let ps = builder.build().unwrap();
        let matches = ps.scan(test_data).unwrap();

        let found = matches
            .iter()
            .any(|m| &test_data[m.start as usize..m.end as usize] == b"TARGET_PATTERN");

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Pattern at position {} NOT FOUND!",
            position
        );
    }
}

/// Regression test: HashScanner boundary (5000 patterns) must not drop patterns
#[test]
fn p0_hash_scanner_boundary_no_pattern_drop() {
    let target = "burpcollaborator%2Enet";
    let test_data = b"GET /callback?domain=burpcollaborator%2Enet HTTP/1.1";

    for count in [4999, 5000, 5001, 5100] {
        let mut builder = PatternSet::builder();
        builder = builder.literal(target);

        // Add filler patterns to reach target count
        for i in 0..(count - 1) {
            builder = builder.literal(&format!("filler{:05}", i));
        }

        let ps = builder.build().unwrap();
        assert_eq!(ps.len(), count, "PatternSet should have {} patterns", count);

        let matches = ps.scan(test_data).unwrap();
        let found = matches
            .iter()
            .any(|m| &test_data[m.start as usize..m.end as usize] == target.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Target pattern '{}' NOT FOUND with {} patterns!\n\
             HashScanner boundary may be causing pattern drops.",
            target, count
        );
    }
}

/// Regression test: MultiMemchr to AhoCorasick transition (8-9 patterns) must not drop patterns
#[test]
fn p0_multimemchr_aho_corasick_transition_no_drop() {
    let target = "burpcollaborator%2Enet"; // 23 bytes, > 16 so won't use MultiMemchr
    let test_data = b"xxx burpcollaborator%2Enet yyy";

    for count in [7, 8, 9, 10, 15, 20] {
        let mut builder = PatternSet::builder();
        builder = builder.literal(target);

        // Add short filler patterns (<= 16 bytes each)
        for i in 0..(count - 1) {
            builder = builder.literal(&format!("fill{:04}", i));
        }

        let ps = builder.build().unwrap();
        let matches = ps.scan(test_data).unwrap();

        let found = matches
            .iter()
            .any(|m| &test_data[m.start as usize..m.end as usize] == target.as_bytes());

        assert!(
            found,
            "CRITICAL P0 REGRESSION: Target pattern NOT FOUND with {} patterns \
             (MultiMemchr/AhoCorasick transition boundary)!",
            count
        );
    }
}
