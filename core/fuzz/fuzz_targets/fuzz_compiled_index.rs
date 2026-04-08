#![no_main]
use libfuzzer_sys::fuzz_target;
use warpstate::CompiledPatternIndex;

fuzz_target!(|data: &[u8]| {
    // Try loading arbitrary bytes as a compiled index.
    // Must not panic — only return clean errors.
    let result = CompiledPatternIndex::load(data);

    if let Ok(index) = result {
        // If loading succeeds, scan must not panic either.
        let _ = index.scan(b"test input data for scanning");
        let _ = index.scan(b"");
        let _ = index.scan(&[0xFF; 1024]);

        // pattern_count and literal_count must not panic
        let _ = index.literal_count();

        // to_pattern_set must not panic
        let _ = index.to_pattern_set();
    }
});
