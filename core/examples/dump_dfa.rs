/// Debug example: scan data and print match details using the public API.
fn main() {
    let patterns = warpstate::PatternSet::builder()
        .literal("he")
        .literal("hello")
        .literal("world")
        .build()
        .unwrap();

    let data = b"hello world";
    let matches = patterns.scan(data).unwrap();

    println!(
        "Input: {:?}",
        std::str::from_utf8(data).unwrap_or("<binary>")
    );
    println!("Patterns: he, hello, world");
    println!("Matches found: {}", matches.len());
    for m in &matches {
        let matched_bytes = &data[m.start as usize..m.end as usize];
        println!(
            "  pattern_id={}, start={}, end={}, text={:?}",
            m.pattern_id,
            m.start,
            m.end,
            std::str::from_utf8(matched_bytes).unwrap_or("<binary>"),
        );
    }

    // Also demonstrate overlapping scan
    let overlapping = patterns.scan_overlapping(data).unwrap();
    println!("\nOverlapping matches: {}", overlapping.len());
    for m in &overlapping {
        let matched_bytes = &data[m.start as usize..m.end as usize];
        println!(
            "  pattern_id={}, start={}, end={}, text={:?}",
            m.pattern_id,
            m.start,
            m.end,
            std::str::from_utf8(matched_bytes).unwrap_or("<binary>"),
        );
    }
}
