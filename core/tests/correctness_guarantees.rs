//! Correctness audit tests for warpstate pattern matching.
//!
//! Each test verifies a specific correctness guarantee.
//! If the engine behavior is wrong, the test FAILS — that's a finding.

use std::time::{Duration, Instant};
use warpstate::{Error, PatternSet, StreamScanner};

// =============================================================================
// 1. OVERLAPPING MATCHES
// =============================================================================

/// When patterns 'abc' and 'bc' both exist, scanning 'abc' must return BOTH matches.
#[test]
fn overlapping_matches_abc_and_bc() {
    let ps = PatternSet::builder()
        .literal("abc")
        .literal("bc")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"abc").unwrap();

    let abc_match = matches
        .iter()
        .find(|m| m.pattern_id == 0 && m.start == 0 && m.end == 3);
    let bc_match = matches
        .iter()
        .find(|m| m.pattern_id == 1 && m.start == 1 && m.end == 3);

    assert!(
        abc_match.is_some(),
        "CRITICAL: Pattern 'abc' not found in overlapping scan of 'abc'. Matches: {:?}",
        matches
    );
    assert!(
        bc_match.is_some(),
        "CRITICAL: Pattern 'bc' not found in overlapping scan of 'abc'. Matches: {:?}",
        matches
    );
}

// =============================================================================
// 2. PATTERN AT BUFFER BOUNDARY (stream mode)
// =============================================================================

/// A pattern that spans two chunks must be found when input is split at a boundary.
#[test]
fn pattern_at_buffer_boundary_stream_mode() {
    let ps = PatternSet::builder().literal("boundary").build().unwrap();
    let mut scanner = StreamScanner::new(&ps).unwrap();

    // Split exactly in the middle: "boun" | "dary"
    let first = scanner.feed(b"boun").unwrap();
    assert!(
        first.is_empty(),
        "First chunk should not contain a complete match"
    );

    let second = scanner.feed(b"dary").unwrap();
    assert_eq!(
        second.len(),
        1,
        "Pattern spanning chunk boundary must be found"
    );
    assert_eq!(second[0].pattern_id, 0);
    assert_eq!(second[0].start, 0);
    assert_eq!(second[0].end, 8);
}

// =============================================================================
// 3. EMPTY PATTERN
// =============================================================================

/// Zero-length patterns must be rejected at build time, not accepted.
#[test]
fn empty_pattern_rejected_at_build_time() {
    let literal_result = PatternSet::builder().literal("").build();
    assert!(
        matches!(literal_result, Err(Error::EmptyPattern { index: 0 })),
        "Empty literal must be rejected at build time"
    );

    let bytes_result = PatternSet::builder().literal_bytes(b"").build();
    assert!(
        matches!(bytes_result, Err(Error::EmptyPattern { index: 0 })),
        "Empty literal bytes must be rejected at build time"
    );

    let regex_result = PatternSet::builder().regex("").build();
    assert!(
        matches!(regex_result, Err(Error::EmptyPattern { index: 0 })),
        "Empty regex must be rejected at build time"
    );
}

// =============================================================================
// 4. REGEX BACKTRACKING / ReDoS
// =============================================================================

/// Pathological regexes like '(a+)+b' must be rejected at build time.
/// Safe regexes must scan ReDoS input without catastrophic slowdown.
#[test]
fn regex_redos_rejected_with_timeout() {
    // The pathological pattern must be rejected.
    let pathological = PatternSet::builder().regex(r"(a+)+b").build();
    assert!(
        matches!(pathological, Err(Error::PathologicalRegex { index: 0 })),
        "Pathological regex '(a+)+b' must be rejected at build time"
    );

    // A safe equivalent must scan adversarial input in well under a second.
    // Classic ReDoS input for NFA: many 'a's with no trailing 'b'.
    let safe = PatternSet::builder().regex(r"a+b").build().unwrap();
    let input = vec![b'a'; 1_000_000];
    let start = Instant::now();
    let matches = safe.scan(&input).unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "Safe regex scan took {:?} — possible ReDoS vulnerability",
        elapsed
    );
    assert!(
        matches.is_empty(),
        "Regex 'a+b' should find no match in input with no 'b'"
    );
}

// =============================================================================
// 5. MAXIMUM PATTERN COUNT
// =============================================================================

/// 100K patterns must build and scan successfully.
#[test]
fn max_pattern_count_100k_builds_and_scans() {
    let mut builder = PatternSet::builder();
    for i in 0..100_000 {
        builder = builder.literal(&format!("p{:08x}", i));
    }
    let ps = builder.build().expect("100K patterns should build");
    assert_eq!(ps.len(), 100_000);

    let input = b"p00000000 p0001869f p000186a0";
    let matches = ps.scan(input).expect("Should scan with 100K patterns");
    assert!(!matches.is_empty(), "Should find matches in 100K pattern set");
}

/// 1M patterns must build without panic or OOM.
#[test]
fn max_pattern_count_1m_does_not_panic() {
    let mut builder = PatternSet::builder();
    for i in 0..1_000_000 {
        builder = builder.literal(&format!("{:08x}", i));
    }
    let result = builder.build();
    // It may succeed or return an explicit error, but it must NOT panic or OOM.
    match result {
        Ok(ps) => {
            assert_eq!(ps.len(), 1_000_000);
            let input = b"00000000 000f4240 ffffffff";
            let matches = ps.scan(input).unwrap();
            assert!(!matches.is_empty(), "Should find matches in 1M pattern set");
        }
        Err(e) => {
            // An explicit limit error is acceptable; panic/OOM is not.
            println!("1M pattern build returned error (acceptable): {:?}", e);
        }
    }
}

// =============================================================================
// 6. UNICODE
// =============================================================================

/// Multi-byte UTF-8 patterns must match as raw bytes.
#[test]
fn unicode_multibyte_match() {
    let ps = PatternSet::builder().literal("🚀").build().unwrap();
    let input = "prefix🚀suffix".as_bytes();
    let matches = ps.scan(input).unwrap();

    assert_eq!(matches.len(), 1, "Unicode literal must match");
    assert_eq!(matches[0].pattern_id, 0);
    assert_eq!(matches[0].start, 6, "Match must start at byte offset 6");
    assert_eq!(
        matches[0].end, 10,
        "Match must end at byte offset 10 (4-byte emoji)"
    );
}

/// Invalid UTF-8 in input must not break scanning; engine treats input as raw bytes.
#[test]
fn invalid_utf8_scanned_as_bytes() {
    let ps = PatternSet::builder().literal("test").build().unwrap();
    let input = vec![b't', b'e', b's', b't', 0xFF, b't', b'e', b's', b't'];
    let matches = ps.scan(&input).unwrap();

    assert_eq!(
        matches.len(),
        2,
        "Must find both literal matches around invalid UTF-8 byte"
    );
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[0].end, 4);
    assert_eq!(matches[1].start, 5);
    assert_eq!(matches[1].end, 9);
}

/// Regex patterns must also treat bytes as raw (unicode disabled).
#[test]
fn regex_matches_invalid_utf8_byte() {
    let ps = PatternSet::builder().regex(r"\xFF").build().unwrap();
    let input = vec![0x00, 0xFF, 0x00];
    let matches = ps.scan(&input).unwrap();

    assert_eq!(matches.len(), 1, "Regex must match invalid UTF-8 byte 0xFF");
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 2);
}

// =============================================================================
// 7. NULL BYTES
// =============================================================================

/// A pattern containing an embedded null byte must match correctly.
#[test]
fn null_bytes_pattern_match() {
    let ps = PatternSet::builder()
        .literal_bytes(b"a\x00b")
        .build()
        .unwrap();

    let input = b"x a\x00b y";
    let matches = ps.scan(input).unwrap();

    assert_eq!(matches.len(), 1, "Pattern with embedded null must match");
    assert_eq!(matches[0].start, 2);
    assert_eq!(matches[0].end, 5);
}

/// Input containing null bytes must be scanned fully.
#[test]
fn null_bytes_input_scanned_fully() {
    let ps = PatternSet::builder().literal("end").build().unwrap();
    let input = b"abc\x00end\x00def";
    let matches = ps.scan(input).unwrap();

    assert_eq!(matches.len(), 1, "Pattern after null byte must be found");
    assert_eq!(matches[0].start, 4);
    assert_eq!(matches[0].end, 7);
}

/// A pattern consisting entirely of null bytes must match.
#[test]
fn all_null_pattern_matches() {
    let ps = PatternSet::builder()
        .literal_bytes(b"\x00\x00\x00")
        .build()
        .unwrap();

    let input = vec![0x00; 5];
    let matches = ps.scan_overlapping(&input).unwrap();

    assert_eq!(
        matches.len(),
        3,
        "All-null pattern should produce 3 overlapping matches"
    );
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 1);
    assert_eq!(matches[2].start, 2);
}

// =============================================================================
// 8. DUPLICATE PATTERNS
// =============================================================================

/// Duplicate patterns must both be reported in overlapping mode.
#[test]
fn duplicate_patterns_overlap_reports_both() {
    let ps = PatternSet::builder()
        .literal("DUPE")
        .literal("DUPE")
        .build()
        .unwrap();

    let matches = ps.scan_overlapping(b"xxDUPEyy").unwrap();

    assert_eq!(
        matches.len(),
        2,
        "Overlapping scan of duplicate patterns must report both pattern IDs"
    );
    let ids: Vec<_> = matches.iter().map(|m| m.pattern_id).collect();
    assert!(ids.contains(&0), "Pattern ID 0 must be present");
    assert!(ids.contains(&1), "Pattern ID 1 must be present");
}

/// Non-overlapping scan reports the first duplicate only (Aho-Corasick dedup).
#[test]
fn duplicate_patterns_non_overlap_does_not_duplicate() {
    let ps = PatternSet::builder()
        .literal("DUPE")
        .literal("DUPE")
        .build()
        .unwrap();

    let matches = ps.scan(b"xxDUPEyy").unwrap();

    // Aho-Corasick find_iter with LeftmostLongest returns only the first pattern
    // at each position, so we expect exactly one match here.
    assert_eq!(
        matches.len(),
        1,
        "Non-overlapping scan should return exactly one match for duplicate literals"
    );
    assert_eq!(matches[0].pattern_id, 0);
    assert_eq!(matches[0].start, 2);
    assert_eq!(matches[0].end, 6);
}
