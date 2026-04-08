use warpstate::PatternSet;

fn main() {
    let patterns = PatternSet::builder()
        .named_literal("password", "password")
        .named_literal("secret", "secret")
        .build()
        .expect("patterns should compile");

    let matches = patterns
        .scan(b"my password is secret")
        .expect("scan should succeed");

    println!("patterns={}", patterns.len());
    println!("matches={}", matches.len());
}
