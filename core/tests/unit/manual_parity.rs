use warpstate::{CompiledPatternIndex, PatternSet};

#[test]
fn test_leftmost_first_parity() {
    let ps = PatternSet::builder()
        .literal("abcde")
        .literal("bc")
        .build()
        .unwrap();

    let data = b"abcde";
    let ps_matches = ps.scan(data).unwrap();

    let serialized = CompiledPatternIndex::build(&ps).unwrap();
    let index = CompiledPatternIndex::load(&serialized).unwrap();
    let index_matches = index.scan(data).unwrap();

    println!("PS matches: {:?}", ps_matches);
    println!("Index matches: {:?}", index_matches);

    assert_eq!(ps_matches, index_matches);
}
