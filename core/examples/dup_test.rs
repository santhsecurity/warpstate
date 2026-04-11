use std::io::Write;
use warpstate::{HashScanner, PatternSet};

fn main() {
    let patterns = PatternSet::builder()
        .literal("dup")
        .literal("dup")
        .build()
        .unwrap();
    let data = b"xxdupxx";

    let scanner = HashScanner::build(patterns.ir()).unwrap();
    let hash_matches = scanner.scan(data);
    let ac_matches = patterns.scan(data).unwrap();

    let mut f = std::fs::File::create("/tmp/dup_test_output.txt").unwrap();
    writeln!(f, "hash_matches: {:?}", hash_matches).unwrap();
    writeln!(f, "ac_matches: {:?}", ac_matches).unwrap();
}
