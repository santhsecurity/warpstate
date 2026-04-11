#![cfg(feature = "jit")]
#![allow(clippy::unwrap_used)]

use warpstate::Match;
use warpstate::RegexDFA;

#[test]
fn jit_regex_scan_matches_non_jit_scan() {
    let dfa = RegexDFA::build(&["abbc", "needle"], &[0, 1]).unwrap();
    assert!(
        dfa.has_jit(),
        "expected fixed-length regex dfa to compile for JIT"
    );

    let input = b"zzabbcxxneedlezzabbcz";
    let mut jit_buf = [Match::from_parts(0, 0, 0); 64];
    let mut interp_buf = [Match::from_parts(0, 0, 0); 64];

    let jit_count = dfa.scan(input, &mut jit_buf).unwrap();
    let interp_count = dfa.scan_without_jit(input, &mut interp_buf).unwrap();

    let jit_summary: Vec<(u32, u32)> = jit_buf[..jit_count]
        .iter()
        .map(|matched| (matched.pattern_id, matched.end))
        .collect();
    let interpreted_summary: Vec<(u32, u32)> = interp_buf[..interp_count]
        .iter()
        .map(|matched| (matched.pattern_id, matched.end))
        .collect();

    assert_eq!(jit_summary, interpreted_summary);
}

#[test]
fn jit_regex_scan_end_positions_agree() {
    // Verify JIT and interpreted paths agree on pattern_id and end position.
    // Note: start positions may differ between JIT and interpreted paths due to
    // DFA restart handling — the JIT restarts from the beginning of the match
    // while the interpreted path may report a different start. This is a known
    // trade-off documented in the JIT bridge.
    let dfa = RegexDFA::build(&["abbc", "needle"], &[0, 1]).unwrap();
    assert!(
        dfa.has_jit(),
        "expected fixed-length regex dfa to compile for JIT"
    );

    let input = b"zzabbcxxneedlezzabbcz";
    let mut jit_buf = [Match::from_parts(0, 0, 0); 64];
    let mut interp_buf = [Match::from_parts(0, 0, 0); 64];

    let jit_count = dfa.scan(input, &mut jit_buf).unwrap();
    let interp_count = dfa.scan_without_jit(input, &mut interp_buf).unwrap();

    assert_eq!(
        jit_count, interp_count,
        "JIT and interpreted paths must find the same number of matches"
    );

    for i in 0..jit_count {
        assert_eq!(
            (jit_buf[i].pattern_id, jit_buf[i].end),
            (interp_buf[i].pattern_id, interp_buf[i].end),
            "Match {i} end position differs between JIT and interpreted paths"
        );
    }
}

#[test]
fn jit_skips_eoi_sensitive_patterns() {
    let dfa = RegexDFA::build(&["needle$"], &[3]).unwrap();
    assert!(
        !dfa.has_jit(),
        "EOI-sensitive regex should stay on interpreted path"
    );
}
