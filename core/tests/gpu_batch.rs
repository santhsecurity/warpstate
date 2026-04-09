use warpstate::algebraic::AlgebraicDfaMatcher;
use warpstate::batch::{scan_batch_gpu, ScanItem};
use warpstate::gpu_smem::SmemDfaMatcher;
use warpstate::matcher::BlockMatcher;
use warpstate::{AutoMatcher, Error, PatternSet};

#[test]
fn scan_batch_gpu_attributes_matches_to_source_items() {
    let patterns = PatternSet::builder().literal("needle").build().unwrap();
    let matcher = pollster::block_on(AutoMatcher::with_config(
        &patterns,
        warpstate::AutoMatcherConfig::new().gpu_threshold(0),
    ))
    .unwrap();
    let items: Vec<Vec<u8>> = (0..100)
        .map(|idx| format!("id-{idx:03}-needle").into_bytes())
        .collect();
    let scan_items: Vec<ScanItem<'_>> = items
        .iter()
        .enumerate()
        .map(|(idx, bytes)| ScanItem {
            id: idx as u64,
            data: bytes,
        })
        .collect();

    let matches = pollster::block_on(scan_batch_gpu(&matcher, scan_items)).unwrap();
    assert_eq!(matches.len(), 100);
    for (idx, matched) in matches.iter().enumerate() {
        assert_eq!(matched.source_id, idx as u64);
        assert_eq!(matched.matched.pattern_id, 0);
        assert_eq!(matched.matched.start, 7);
    }
}

#[test]
fn scan_batch_gpu_rejects_cross_item_matches_across_100_items() {
    let patterns = PatternSet::builder().literal("needle").build().unwrap();
    let matcher = pollster::block_on(AutoMatcher::with_config(
        &patterns,
        warpstate::AutoMatcherConfig::new().gpu_threshold(0),
    ))
    .unwrap();

    let items: Vec<Vec<u8>> = (0..100)
        .map(|idx| match idx {
            49 => b"nee".to_vec(),
            50 => b"dle".to_vec(),
            _ => format!("item-{idx:03}-needle").into_bytes(),
        })
        .collect();
    let scan_items: Vec<ScanItem<'_>> = items
        .iter()
        .enumerate()
        .map(|(idx, bytes)| ScanItem {
            id: idx as u64,
            data: bytes,
        })
        .collect();

    let matches = pollster::block_on(scan_batch_gpu(&matcher, scan_items)).unwrap();
    assert_eq!(matches.len(), 98);
    assert!(!matches
        .iter()
        .any(|matched| matched.source_id == 49 || matched.source_id == 50));
}

#[test]
fn algebraic_matcher_handles_non_matching_data() {
    let patterns = PatternSet::builder()
        .regex("needle|needlf")
        .build()
        .unwrap();
    let matcher = match pollster::block_on(AlgebraicDfaMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected algebraic init error: {other:?}"),
    };

    let data = b"xxnomatchyy";
    let gpu_matches = pollster::block_on(matcher.scan_block(data)).unwrap();
    let cpu_matches = patterns.scan(data).unwrap();
    assert_eq!(gpu_matches, cpu_matches);
}

#[test]
fn smem_matcher_scans_known_data() {
    let patterns = PatternSet::builder()
        .regex("needle|needlf")
        .build()
        .unwrap();
    let matcher = match pollster::block_on(SmemDfaMatcher::new(&patterns)) {
        Ok(matcher) => matcher,
        Err(Error::NoGpuAdapter) => return,
        Err(other) => panic!("unexpected smem init error: {other:?}"),
    };

    let matches = pollster::block_on(matcher.scan_block(b"xxneedleyy")).unwrap();
    assert!(!matches.is_empty());
    assert_eq!(matches[0].start, 2);
    assert_eq!(matches[0].end, 8);
}
