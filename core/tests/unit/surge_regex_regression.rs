use std::path::PathBuf;

use warpstate::PatternSet;

const SURGE_MEMBER_ACCESS: &str = r"\b[a-zA-Z_]\w*\s*->\s*[a-zA-Z_]\w*\b";

fn linux_vpe_fixture() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../../corpus/repos/linux/arch/mips/kernel/vpe.c");
    std::fs::read(path).expect("linux vpe.c fixture must be readable")
}

fn assert_member_access_matches(patterns: &PatternSet, data: &[u8], pattern_id: u32) {
    let matches = patterns.scan(data).expect("scan must succeed");
    let hits = matches
        .iter()
        .filter(|matched| matched.pattern_id == pattern_id)
        .count();
    assert_eq!(hits, 100, "SURGE member-access regex must match vpe.c");
}

#[test]
fn surge_member_access_regex_matches_linux_fixture() {
    let data = linux_vpe_fixture();
    let patterns = PatternSet::builder()
        .regex(SURGE_MEMBER_ACCESS)
        .build()
        .expect("SURGE regex must compile");

    assert_member_access_matches(&patterns, &data, 0);
}

#[test]
fn surge_member_access_regex_survives_large_pattern_set() {
    let data = linux_vpe_fixture();
    let mut builder = PatternSet::builder();

    for index in 0..855 {
        builder = builder.literal_bytes(format!("unlikely_literal_{index:03}").into_bytes());
    }
    builder = builder.regex(SURGE_MEMBER_ACCESS);

    let patterns = builder
        .build()
        .expect("large SURGE PatternSet must compile");
    assert_eq!(patterns.len(), 856);
    assert_member_access_matches(&patterns, &data, 855);
}

#[test]
fn byte_regex_fast_path_uses_dfa_compatible_syntax() {
    let patterns = PatternSet::builder()
        .regex(r"(?-u:\xFF)")
        .build()
        .expect("byte regex must compile");

    let matches = patterns
        .scan(&[0x00, 0xFF, 0x01])
        .expect("scan must succeed");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 2);
}

#[test]
fn overlapping_scan_grows_past_initial_capacity() {
    let mut builder = PatternSet::builder();
    for _ in 0..10 {
        builder = builder.literal_bytes(b"a");
    }
    let patterns = builder.build().expect("duplicate literals must compile");
    let data = vec![b'a'; 2_000];

    let matches = patterns
        .scan_overlapping(&data)
        .expect("overlapping scan must grow the match buffer");
    assert_eq!(matches.len(), 20_000);
}
