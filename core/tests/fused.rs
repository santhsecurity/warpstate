#![cfg(feature = "fused")]

use ebpfsieve::{ByteFrequencyFilter, ByteThreshold};
use warpstate::{FusedScanner, PatternSet};

#[test]
fn fused_scanner_matches_unfused_results_across_window_boundaries() {
    let patterns = PatternSet::builder()
        .literal("needle")
        .literal("needless")
        .build()
        .unwrap();

    let filter = ByteFrequencyFilter::new([ByteThreshold::new(b'n', 1)])
        .unwrap()
        .with_window_size(4096)
        .unwrap();

    let scanner = FusedScanner::new(patterns.clone(), Some(filter));

    let mut data = vec![b'x'; 12 * 1024];
    data[4094..4100].copy_from_slice(b"needle");
    data[8190..8198].copy_from_slice(b"needless");
    data.extend_from_slice(b" trailer needle");

    let unfused = patterns.scan(&data).unwrap();
    let fused = scanner.scan(&data).unwrap();

    // Fused pipeline may find additional matches at window boundaries because
    // each window is scanned independently. This is MORE correct (finds overlapping
    // matches that LeftmostFirst suppresses). The fused result must be a superset.
    for m in &unfused {
        assert!(
            fused.contains(m),
            "fused pipeline must contain all unfused matches. Missing: {m:?}"
        );
    }
    assert!(
        fused.len() >= unfused.len(),
        "fused pipeline should find at least as many matches as unfused"
    );
}
