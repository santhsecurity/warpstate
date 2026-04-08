#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 { return; }
    
    // Use the first byte as a split point length mapped to remaining bytes
    let split_len = (data[0] as usize) % data.len();
    if split_len == 0 { return; }
    
    let pat_bytes = &data[1..split_len];
    let haystack = &data[split_len..];

    if let Ok(pat_str) = std::str::from_utf8(pat_bytes) {
        if let Ok(ps) = warpstate::PatternSet::builder().regex(pat_str).build() {
            let _ = ps.scan(haystack);
        }
    }
});
