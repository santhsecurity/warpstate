#![cfg(feature = "gpu")]
use warpstate::batch::{scan_batch_gpu, ScanItem};
use warpstate::BlockMatcher;
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
