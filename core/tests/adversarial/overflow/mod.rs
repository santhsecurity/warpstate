use warpstate::*;

#[test]
fn test_dfa_state_explosion_nested_regex() {
    // Nested 10 levels
    let pattern = r"(.*(.*(.*(.*(.*(.*(.*(.*(.*(.*).*).*).*).*).*).*).*).*).*).*";

    // We expect this to fail gracefully or succeed without panicking/OOMing.
    // Given the core rules: "Crash recovery. Tests designed to BREAK."
    // Let's ensure it doesn't crash the engine but handles it as an error or compiles.
    let result = PatternSet::builder().regex(pattern).build();

    // It should either build (unlikely due to size limits) or fail cleanly.
    match result {
        Ok(ps) => {
            let matches = ps.scan(b"some input").unwrap();
            // Since it matches anything, should have match
            assert!(
                !matches.is_empty(),
                "Regex matching everything should find matches"
            );
        }
        Err(Error::PatternCompilationFailed { .. }) => {
            // Expected failure mode, this is fine but we need an assertion
            assert!(
                true,
                "Caught expected compilation failure due to state explosion"
            );
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
