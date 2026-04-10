use super::CompiledPatternIndex;
use crate::{Error, PatternSet};

#[test]
fn round_trips_literal_and_regex_indexes() {
    let patterns = PatternSet::builder()
        .literal("alpha")
        .regex("b[0-9]+")
        .literal("beta")
        .build()
        .unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let data = b"alpha b42 beta";
    assert_eq!(index.scan(data).unwrap(), patterns.scan(data).unwrap());
}

#[test]
fn preserves_ascii_case_insensitive_literals() {
    let patterns = PatternSet::builder()
        .case_insensitive(true)
        .literal("Needle")
        .build()
        .unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    assert_eq!(
        index.scan(b"xxneedlexx").unwrap(),
        patterns.scan(b"xxneedlexx").unwrap()
    );
}

#[test]
fn rejects_invalid_headers() {
    let err = CompiledPatternIndex::load(b"not-an-index").unwrap_err();
    assert!(matches!(err, Error::PatternCompilationFailed { .. }));
}

#[test]
fn saves_and_loads_from_file() {
    let patterns = PatternSet::builder()
        .literal("disk")
        .regex("load")
        .build()
        .unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let path = std::env::temp_dir().join(format!(
        "warpstate-compiled-index-{}.idx",
        std::process::id()
    ));
    index.save_to_file(&path).unwrap();
    let loaded = CompiledPatternIndex::load_from_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let data = b"disk load";
    assert_eq!(loaded.scan(data).unwrap(), patterns.scan(data).unwrap());
}

#[test]
fn to_pattern_set_rebuilds_full_pattern_set() {
    let patterns = PatternSet::builder()
        .literal("alpha")
        .named_regex("key", r"[A-Z]{3}-\d+")
        .build()
        .unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let path = std::env::temp_dir().join(format!(
        "warpstate-compiled-index-pattern-set-{}.idx",
        std::process::id()
    ));
    index.save_to_file(&path).unwrap();
    let loaded = CompiledPatternIndex::load_from_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let rebuilt = loaded.to_pattern_set().unwrap();
    let data = b"alpha ABC-1234 beta";
    assert_eq!(rebuilt.scan(data).unwrap(), patterns.scan(data).unwrap());
}

#[test]
fn named_patterns_survive_round_trip() {
    let patterns = PatternSet::builder()
        .named_literal("cred", "password")
        .named_regex("key", r"[A-Z]{4}-\d{4}")
        .build()
        .unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let data = b"password ABCD-1234";
    assert_eq!(index.scan(data).unwrap().len(), 2);
}

#[test]
fn many_literals_round_trip() {
    let mut builder = PatternSet::builder();
    for i in 0..100 {
        builder = builder.literal(&format!("pattern_{i:03}"));
    }
    let patterns = builder.build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let data = b"xxpattern_042xxpattern_099xx";
    let expected = patterns.scan(data).unwrap();
    let actual = index.scan(data).unwrap();
    assert_eq!(actual.len(), expected.len());
}

#[test]
fn truncated_index_rejected() {
    let patterns = PatternSet::builder().literal("test").build().unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let err = CompiledPatternIndex::load(&bytes[..bytes.len() / 2]).unwrap_err();
    assert!(matches!(err, Error::PatternCompilationFailed { .. }));
}

#[test]
fn regex_line_anchors_survive_round_trip() {
    let patterns = PatternSet::builder()
        .regex("^fn main$")
        .literal("mod")
        .build()
        .unwrap();
    let bytes = CompiledPatternIndex::build(&patterns).unwrap();
    let index = CompiledPatternIndex::load(&bytes).unwrap();
    let data = b"mod demo;\nfn main\nfn main()\n";
    assert_eq!(index.scan(data).unwrap(), patterns.scan(data).unwrap());
}
