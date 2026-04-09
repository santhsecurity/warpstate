use std::sync::Arc;

use warpstate::PatternSet;


#[test]
fn test_01_build_pattern_set_892_literals() {
    let mut builder = PatternSet::builder();
    let mut input = String::new();
    for i in 0..892 {
        builder = builder.literal(format!("keyhog_pattern_{:04}", i));
        input.push_str(&format!("keyhog_pattern_{:04} ", i));
    }
    let set = builder.build().unwrap();
    
    let matches = set.scan(input.as_bytes()).unwrap();
    
    assert_eq!(matches.len(), 892);
    for i in 0..892 {
        assert_eq!(matches[i].pattern_id, i as u32);
    }
}
#[test]
fn test_02_empty_pattern_set() {
    let set = PatternSet::builder().build().unwrap();
    let matches = set.scan(b"test").unwrap();
    assert_eq!(matches.len(), 0);
}
#[test]
fn test_03_single_1_char_pattern() {
    let set = PatternSet::builder().literal("a").build().unwrap();
    let matches = set.scan(b"xayaza").unwrap();
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[1].start, 3);
    assert_eq!(matches[2].start, 5);
}

#[test]
fn test_04_overlapping_prefix() {
    let set = PatternSet::builder()
        .literal("overlap")
        .literal("over")
        .build().unwrap();
    
    // Non-overlapping scan returns only the first match (longest or first defined depending on AC implementation)
    // For AC, it usually returns the earliest match in overlapping, but non-overlapping depends on strategy.
    // The requirement says "BOTH match (overlapping)". We should use `scan_overlapping`.
    let matches = set.scan_overlapping(b"overlap").unwrap();
    
    // Let's assert what they are. 
    assert_eq!(matches.len(), 2);
    // Usually "over" matches at 0..4, and "overlap" at 0..7
    // Let's print to see or just assert they exist.
    let mut found_over = false;
    let mut found_overlap = false;
    for m in matches {
        if m.pattern_id == 0 && m.start == 0 && m.end == 7 {
            found_overlap = true;
        }
        if m.pattern_id == 1 && m.start == 0 && m.end == 4 {
            found_over = true;
        }
    }
    assert!(found_over);
    assert!(found_overlap);
}

#[test]
fn test_05_case_sensitive() {
    let set = PatternSet::builder().literal("sk_live").build().unwrap();
    let matches = set.scan(b"sk_live SK_LIVE sK_lIvE").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 0);
}

#[test]
fn test_06_scan_empty_input() {
    let set = PatternSet::builder().literal("a").build().unwrap();
    let matches = set.scan(b"").unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_07_scan_1_byte_input() {
    let set = PatternSet::builder().literal("a").build().unwrap();
    let matches = set.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1);
    
    let matches2 = set.scan(b"b").unwrap();
    assert_eq!(matches2.len(), 0);
}

#[test]
fn test_08_scan_100mb_input() {
    let set = PatternSet::builder().literal("needle").build().unwrap();
    let mut input = vec![b'a'; 100 * 1024 * 1024];
    input.extend_from_slice(b"needle");
    
    let matches = set.scan(&input).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start as usize, 100 * 1024 * 1024);
}

#[test]
fn test_09_exponential_blowup_prevention() {
    let set = PatternSet::builder().literal("a").build().unwrap();
    let input = vec![b'a'; 10_000]; // Every byte matches
    let matches = set.scan(&input).unwrap();
    assert_eq!(matches.len(), 10_000);
}

#[test]
fn test_10_pattern_with_null_bytes() {
    let set = PatternSet::builder().literal_bytes(b"a\0b").build().unwrap();
    let matches = set.scan(b"xa\0by").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 1);
    assert_eq!(matches[0].end, 4);
}

#[test]
fn test_11_input_with_null_bytes() {
    let set = PatternSet::builder().literal("needle").build().unwrap();
    let matches = set.scan(b"\0\0needle\0\0").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 2);
}

#[test]
fn test_12_unicode_patterns() {
    let set = PatternSet::builder().literal("🔥").build().unwrap();
    let matches = set.scan("xyz🔥abc".as_bytes()).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 3);
}

#[test]
fn test_13_10000_patterns() {
    let start = std::time::Instant::now();
    let mut builder = PatternSet::builder();
    for i in 0..10_000 {
        builder = builder.literal(format!("keyhog_pattern_{:05}", i));
    }
    let set = builder.build().unwrap();
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 1, "Built in {:?}", elapsed);
    
    let matches = set.scan(b"keyhog_pattern_09999").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pattern_id, 9999);
}

#[test]
fn test_14_duplicate_patterns() {
    // Should be handled properly (both return matches, or one overlaps)
    let set = PatternSet::builder()
        .literal("dup")
        .literal("dup")
        .build().unwrap();
        
    let matches = set.scan_overlapping(b"dup").unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].pattern_id, 0);
    assert_eq!(matches[1].pattern_id, 1);
}

#[test]
fn test_15_very_long_pattern() {
    let long_pattern = "a".repeat(10_000);
    let set = PatternSet::builder().literal(long_pattern.clone()).build();
    // It should either build or error gracefully
    if let Ok(set) = set {
        let mut input = "b".repeat(10_000);
        input.push_str(&long_pattern);
        let matches = set.scan(input.as_bytes()).unwrap();
        assert_eq!(matches.len(), 1);
    } else {
        // Expected an error
    }
}

#[test]
fn test_16_scan_after_builder_error() {
    let set_err = PatternSet::builder().literal("").build();
    assert!(set_err.is_err());
    
    let set_ok = PatternSet::builder().literal("ok").build().unwrap();
    let matches = set_ok.scan(b"ok").unwrap();
    assert_eq!(matches.len(), 1);
}
#[test]
fn test_17_thread_safety() {
    
    let set = Arc::new(PatternSet::builder().literal("thread").build().unwrap());
    
    let mut threads = vec![];
    for _ in 0..10 {
        let set = Arc::clone(&set);
        threads.push(std::thread::spawn(move || {
            let matches = set.scan(b"thread_safe").unwrap();
            assert_eq!(matches.len(), 1);
        }));
    }
    
    for t in threads {
        t.join().unwrap();
    }
}

fn assert_send_sync<T: Send + Sync>() {}
#[test]
fn test_18_pattern_set_send_sync() {
    assert_send_sync::<PatternSet>();
}

#[test]
fn test_19_match_positions_correct() {
    let set = PatternSet::builder().literal("match").build().unwrap();
    let matches = set.scan(b"01234match90123").unwrap();
    assert_eq!(matches[0].start, 5);
    assert_eq!(matches[0].end, 10);
}

#[test]
fn test_20_match_ids_insertion_order() {
    let set = PatternSet::builder()
        .literal("a")
        .literal("b")
        .literal("c")
        .build().unwrap();
    let matches = set.scan(b"c b a").unwrap();
    // They are returned in sorted start offset order:
    // start 0 is "c", ID 2
    // start 2 is "b", ID 1
    // start 4 is "a", ID 0
    assert_eq!(matches[0].pattern_id, 2);
    assert_eq!(matches[1].pattern_id, 1);
    assert_eq!(matches[2].pattern_id, 0);
}

#[test]
fn test_21_scan_twice_deterministic() {
    let set = PatternSet::builder().literal("det").build().unwrap();
    let m1 = set.scan(b"deterministic").unwrap();
    let m2 = set.scan(b"deterministic").unwrap();
    assert_eq!(m1, m2);
}

#[test]
fn test_22_build_with_regex() {
    let set = PatternSet::builder().regex("r[eE]gex").build().unwrap();
    let matches = set.scan(b"regex and rEgex").unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 10);
}

#[test]
fn test_23_mix_literal_and_regex() {
    let set = PatternSet::builder()
        .literal("literal")
        .regex("[0-9]{3}")
        .build().unwrap();
        
    let matches = set.scan(b"literal 123").unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].pattern_id, 0);
    assert_eq!(matches[1].pattern_id, 1);
}

#[test]
fn test_24_pattern_matches_every_position() {
    let set = PatternSet::builder().literal("a").build().unwrap();
    let matches = set.scan_overlapping(b"aaaaa").unwrap();
    assert_eq!(matches.len(), 5);
}

#[test]
fn test_25_common_prefix() {
    let set = PatternSet::builder()
        .literal("sk_live_123")
        .literal("sk_test_456")
        .literal("sk_live_abc")
        .build().unwrap();
        
    let matches = set.scan(b"sk_live_abc and sk_test_456").unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].pattern_id, 2);
    assert_eq!(matches[1].pattern_id, 1);
}

#[test]
fn test_26_simd_boundaries() {
    let set = PatternSet::builder().literal("boundary").build().unwrap();
    // 32 bytes
    let mut input32 = vec![b'x'; 32];
    input32[24..32].copy_from_slice(b"boundary");
    let m1 = set.scan(&input32).unwrap();
    assert_eq!(m1.len(), 1);
    assert_eq!(m1[0].start, 24);
    
    // 64 bytes
    let mut input64 = vec![b'y'; 64];
    input64[56..64].copy_from_slice(b"boundary");
    let m2 = set.scan(&input64).unwrap();
    assert_eq!(m2.len(), 1);
    assert_eq!(m2[0].start, 56);
}

#[test]
fn test_27_all_zeros() {
    let set = PatternSet::builder().literal_bytes(vec![0, 0, 0]).build().unwrap();
    let input = vec![0; 10];
    let matches = set.scan_overlapping(&input).unwrap();
    assert_eq!(matches.len(), 8);
}

#[test]
fn test_28_all_ff() {
    let set = PatternSet::builder().literal_bytes(vec![0xFF, 0xFF]).build().unwrap();
    let input = vec![0xFF; 5];
    let matches = set.scan_overlapping(&input).unwrap();
    assert_eq!(matches.len(), 4);
}

#[test]
fn test_29_stress_test_10000_scans() {
    let set = PatternSet::builder().literal("needle").build().unwrap();
    for i in 0..10_000 {
        let input = format!("haystack_{}_needle_{}", i, i);
        let matches = set.scan(input.as_bytes()).unwrap();
        assert_eq!(matches.len(), 1);
    }
}

#[test]
fn test_30_memory_drop() {
    // We create and drop many sets to ensure no leaks or crashes
    for _ in 0..100 {
        let _set = PatternSet::builder().literal("dropme").build().unwrap();
    }
}

#[test]
fn test_31_clone() {
    let set = PatternSet::builder().literal("clone_test").build().unwrap();
    let set_2 = set.clone();
    
    let m1 = set.scan(b"clone_test").unwrap();
    let m2 = set_2.scan(b"clone_test").unwrap();
    assert_eq!(m1, m2);
}
#[test]
fn test_32_serialize_deserialize_index() {
    let set = PatternSet::builder()
        .literal("index_test")
        .build().unwrap();
        
    let index_bytes = warpstate::CompiledPatternIndex::build(&set).unwrap();
    let loaded_index = warpstate::CompiledPatternIndex::load(&index_bytes).unwrap();
    
    let matches = loaded_index.scan(b"this is an index_test").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 11);
}

#[test]
fn test_33_scan_gpu_disabled() {
    // Just a standard scan, which shouldn't use GPU unless explicitly requested
    let set = PatternSet::builder().literal("nogpu").build().unwrap();
    let matches = set.scan(b"test nogpu test").unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].start, 5);
}

#[test]
fn test_34_scan_overlapping_vs_non_overlapping() {
    let set = PatternSet::builder().literal("ana").build().unwrap();
    
    let non_overlap = set.scan(b"banana").unwrap();
    assert_eq!(non_overlap.len(), 1); // Only the first 'ana' starting at index 1 is found
    
    let overlap = set.scan_overlapping(b"banana").unwrap();
    assert_eq!(overlap.len(), 2); // 'ana' starting at 1, 'ana' starting at 3
}

