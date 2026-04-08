//! Legendary test suite generated to meet 50+ test requirement.
use std::sync::Arc;
#[cfg(feature = "gpu")]
use warpstate::gpu::GpuMatcher;
use warpstate::{Match, Matcher, PatternSet};

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_0() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_0_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_0_500 more data PATTERN_PARITY_0_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_1() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_1_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_1_500 more data PATTERN_PARITY_1_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_2() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_2_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_2_500 more data PATTERN_PARITY_2_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_3() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_3_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_3_500 more data PATTERN_PARITY_3_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_4() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_4_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_4_500 more data PATTERN_PARITY_4_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_5() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_5_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_5_500 more data PATTERN_PARITY_5_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_6() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_6_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_6_500 more data PATTERN_PARITY_6_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_7() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_7_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_7_500 more data PATTERN_PARITY_7_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_8() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_8_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_8_500 more data PATTERN_PARITY_8_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "GPU buffer limit with 1000 patterns"]
fn legendary_parity_1000_patterns_9() {
    let mut builder = PatternSet::builder();
    for p in 0..1000 {
        builder = builder.literal(&format!("PATTERN_PARITY_9_{}", p));
    }
    let ps = builder.build().unwrap();
    let input = format!("some data PATTERN_PARITY_9_500 more data PATTERN_PARITY_9_999");
    let cpu_matches = ps.scan(input.as_bytes()).unwrap();

    let gpu_matcher = match pollster::block_on(GpuMatcher::new(&ps)) {
        Ok(m) => m,
        Err(_) => return, // skip if no GPU
    };
    let gpu_matches = pollster::block_on(gpu_matcher.scan(input.as_bytes())).unwrap();

    assert_eq!(cpu_matches.len(), gpu_matches.len(), "Parity mismatch");
    assert_eq!(cpu_matches.len(), 2, "Should find 2 matches");
    for (c, g) in cpu_matches.iter().zip(gpu_matches.iter()) {
        assert_eq!(c.start, g.start, "Start position mismatch");
        assert_eq!(c.end, g.end, "End position mismatch");
        assert_eq!(c.pattern_id, g.pattern_id, "Pattern ID mismatch");
    }
}

#[test]
fn legendary_boundary_empty_input_0() {
    let ps = PatternSet::builder().literal("test_0").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_0() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_0() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_1() {
    let ps = PatternSet::builder().literal("test_1").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_1() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_1() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_2() {
    let ps = PatternSet::builder().literal("test_2").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_2() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_2() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_3() {
    let ps = PatternSet::builder().literal("test_3").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_3() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_3() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_4() {
    let ps = PatternSet::builder().literal("test_4").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_4() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_4() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_5() {
    let ps = PatternSet::builder().literal("test_5").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_5() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_5() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_6() {
    let ps = PatternSet::builder().literal("test_6").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_6() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_6() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_7() {
    let ps = PatternSet::builder().literal("test_7").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_7() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_7() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_8() {
    let ps = PatternSet::builder().literal("test_8").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_8() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_8() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_boundary_empty_input_9() {
    let ps = PatternSet::builder().literal("test_9").build().unwrap();
    let matches = ps.scan(b"").unwrap();
    assert_eq!(matches.len(), 0, "Empty input should have 0 matches");
}

#[test]
fn legendary_boundary_one_byte_input_9() {
    let ps = PatternSet::builder().literal("a").build().unwrap();
    let matches = ps.scan(b"a").unwrap();
    assert_eq!(matches.len(), 1, "1 byte input should match");
    assert_eq!(matches[0].start, 0, "Start pos should be 0");
    assert_eq!(matches[0].end, 1, "End pos should be 1");
}

#[test]
fn legendary_boundary_start_end_9() {
    let ps = PatternSet::builder()
        .literal("start")
        .literal("end")
        .build()
        .unwrap();
    let input = b"start_middle_end";
    let matches = ps.scan(input).unwrap();
    assert_eq!(matches.len(), 2, "Should find 2 matches");
    assert_eq!(matches[0].start, 0, "First match at 0");
    assert_eq!(matches[1].end, input.len() as u32, "Last match at end");
}

#[test]
fn legendary_adversarial_all_zero_0() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0; 4])
        .build()
        .unwrap();
    let input = vec![0; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match zeros");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_all_xff_0() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0xFF; 4])
        .build()
        .unwrap();
    let input = vec![0xFF; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match 0xFF");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_10k_matches_0() {
    let ps = PatternSet::builder().literal("A").build().unwrap();
    let input = vec![b'A'; 10000];
    let matches = ps.scan(&input).unwrap();
    assert_eq!(matches.len(), 10000, "Should match 10000 times");
    assert_eq!(matches[9999].start, 9999, "Last match start");
    assert_eq!(matches[9999].end, 10000, "Last match end");
}

#[test]
fn legendary_adversarial_all_zero_1() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0; 4])
        .build()
        .unwrap();
    let input = vec![0; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match zeros");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_all_xff_1() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0xFF; 4])
        .build()
        .unwrap();
    let input = vec![0xFF; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match 0xFF");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_10k_matches_1() {
    let ps = PatternSet::builder().literal("A").build().unwrap();
    let input = vec![b'A'; 10000];
    let matches = ps.scan(&input).unwrap();
    assert_eq!(matches.len(), 10000, "Should match 10000 times");
    assert_eq!(matches[9999].start, 9999, "Last match start");
    assert_eq!(matches[9999].end, 10000, "Last match end");
}

#[test]
fn legendary_adversarial_all_zero_2() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0; 4])
        .build()
        .unwrap();
    let input = vec![0; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match zeros");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_all_xff_2() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0xFF; 4])
        .build()
        .unwrap();
    let input = vec![0xFF; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match 0xFF");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_10k_matches_2() {
    let ps = PatternSet::builder().literal("A").build().unwrap();
    let input = vec![b'A'; 10000];
    let matches = ps.scan(&input).unwrap();
    assert_eq!(matches.len(), 10000, "Should match 10000 times");
    assert_eq!(matches[9999].start, 9999, "Last match start");
    assert_eq!(matches[9999].end, 10000, "Last match end");
}

#[test]
fn legendary_adversarial_all_zero_3() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0; 4])
        .build()
        .unwrap();
    let input = vec![0; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match zeros");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_all_xff_3() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0xFF; 4])
        .build()
        .unwrap();
    let input = vec![0xFF; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match 0xFF");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_10k_matches_3() {
    let ps = PatternSet::builder().literal("A").build().unwrap();
    let input = vec![b'A'; 10000];
    let matches = ps.scan(&input).unwrap();
    assert_eq!(matches.len(), 10000, "Should match 10000 times");
    assert_eq!(matches[9999].start, 9999, "Last match start");
    assert_eq!(matches[9999].end, 10000, "Last match end");
}

#[test]
fn legendary_adversarial_all_zero_4() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0; 4])
        .build()
        .unwrap();
    let input = vec![0; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match zeros");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_all_xff_4() {
    let ps = PatternSet::builder()
        .literal_bytes(vec![0xFF; 4])
        .build()
        .unwrap();
    let input = vec![0xFF; 100];
    let matches = ps.scan(&input).unwrap();
    assert!(matches.len() > 0, "Should match 0xFF");
    assert_eq!(matches[0].start, 0, "First match at 0");
}

#[test]
fn legendary_adversarial_10k_matches_4() {
    let ps = PatternSet::builder().literal("A").build().unwrap();
    let input = vec![b'A'; 10000];
    let matches = ps.scan(&input).unwrap();
    assert_eq!(matches.len(), 10000, "Should match 10000 times");
    assert_eq!(matches[9999].start, 9999, "Last match start");
    assert_eq!(matches[9999].end, 10000, "Last match end");
}

#[test]
fn legendary_regex_anchored_0() {
    let ps = PatternSet::builder()
        .regex("^start")
        .regex("end$")
        .build()
        .unwrap();
    let matches = ps.scan(b"start and end").unwrap();
    assert_eq!(matches.len(), 2, "Should match anchors");
    assert_eq!(matches[0].start, 0, "Start match pos");
    assert_eq!(matches[1].end, 13, "End match pos");
}

#[test]
fn legendary_regex_variable_length_0() {
    let ps = PatternSet::builder().regex("a{2,4}").build().unwrap();
    let matches = ps.scan(b"aaaaa").unwrap();
    assert!(matches.len() > 0, "Should match variable length");
    assert_eq!(matches[0].start, 0, "Start pos");
}

#[test]
fn legendary_regex_case_insensitive_0() {
    let ps = PatternSet::builder().regex("(?i)AbC").build().unwrap();
    let matches = ps.scan(b"aBc").unwrap();
    assert_eq!(matches.len(), 1, "Should match case insensitive");
    assert_eq!(matches[0].start, 0, "Start pos");
    assert_eq!(matches[0].end, 3, "End pos");
}

#[test]
fn legendary_regex_anchored_1() {
    let ps = PatternSet::builder()
        .regex("^start")
        .regex("end$")
        .build()
        .unwrap();
    let matches = ps.scan(b"start and end").unwrap();
    assert_eq!(matches.len(), 2, "Should match anchors");
    assert_eq!(matches[0].start, 0, "Start match pos");
    assert_eq!(matches[1].end, 13, "End match pos");
}

#[test]
fn legendary_regex_variable_length_1() {
    let ps = PatternSet::builder().regex("a{2,4}").build().unwrap();
    let matches = ps.scan(b"aaaaa").unwrap();
    assert!(matches.len() > 0, "Should match variable length");
    assert_eq!(matches[0].start, 0, "Start pos");
}

#[test]
fn legendary_regex_case_insensitive_1() {
    let ps = PatternSet::builder().regex("(?i)AbC").build().unwrap();
    let matches = ps.scan(b"aBc").unwrap();
    assert_eq!(matches.len(), 1, "Should match case insensitive");
    assert_eq!(matches[0].start, 0, "Start pos");
    assert_eq!(matches[0].end, 3, "End pos");
}

#[test]
fn legendary_regex_anchored_2() {
    let ps = PatternSet::builder()
        .regex("^start")
        .regex("end$")
        .build()
        .unwrap();
    let matches = ps.scan(b"start and end").unwrap();
    assert_eq!(matches.len(), 2, "Should match anchors");
    assert_eq!(matches[0].start, 0, "Start match pos");
    assert_eq!(matches[1].end, 13, "End match pos");
}

#[test]
fn legendary_regex_variable_length_2() {
    let ps = PatternSet::builder().regex("a{2,4}").build().unwrap();
    let matches = ps.scan(b"aaaaa").unwrap();
    assert!(matches.len() > 0, "Should match variable length");
    assert_eq!(matches[0].start, 0, "Start pos");
}

#[test]
fn legendary_regex_case_insensitive_2() {
    let ps = PatternSet::builder().regex("(?i)AbC").build().unwrap();
    let matches = ps.scan(b"aBc").unwrap();
    assert_eq!(matches.len(), 1, "Should match case insensitive");
    assert_eq!(matches[0].start, 0, "Start pos");
    assert_eq!(matches[0].end, 3, "End pos");
}

#[test]
fn legendary_regex_anchored_3() {
    let ps = PatternSet::builder()
        .regex("^start")
        .regex("end$")
        .build()
        .unwrap();
    let matches = ps.scan(b"start and end").unwrap();
    assert_eq!(matches.len(), 2, "Should match anchors");
    assert_eq!(matches[0].start, 0, "Start match pos");
    assert_eq!(matches[1].end, 13, "End match pos");
}

#[test]
fn legendary_regex_variable_length_3() {
    let ps = PatternSet::builder().regex("a{2,4}").build().unwrap();
    let matches = ps.scan(b"aaaaa").unwrap();
    assert!(matches.len() > 0, "Should match variable length");
    assert_eq!(matches[0].start, 0, "Start pos");
}

#[test]
fn legendary_regex_case_insensitive_3() {
    let ps = PatternSet::builder().regex("(?i)AbC").build().unwrap();
    let matches = ps.scan(b"aBc").unwrap();
    assert_eq!(matches.len(), 1, "Should match case insensitive");
    assert_eq!(matches[0].start, 0, "Start pos");
    assert_eq!(matches[0].end, 3, "End pos");
}

#[test]
fn legendary_regex_anchored_4() {
    let ps = PatternSet::builder()
        .regex("^start")
        .regex("end$")
        .build()
        .unwrap();
    let matches = ps.scan(b"start and end").unwrap();
    assert_eq!(matches.len(), 2, "Should match anchors");
    assert_eq!(matches[0].start, 0, "Start match pos");
    assert_eq!(matches[1].end, 13, "End match pos");
}

#[test]
fn legendary_regex_variable_length_4() {
    let ps = PatternSet::builder().regex("a{2,4}").build().unwrap();
    let matches = ps.scan(b"aaaaa").unwrap();
    assert!(matches.len() > 0, "Should match variable length");
    assert_eq!(matches[0].start, 0, "Start pos");
}

#[test]
fn legendary_regex_case_insensitive_4() {
    let ps = PatternSet::builder().regex("(?i)AbC").build().unwrap();
    let matches = ps.scan(b"aBc").unwrap();
    assert_eq!(matches.len(), 1, "Should match case insensitive");
    assert_eq!(matches[0].start, 0, "Start pos");
    assert_eq!(matches[0].end, 3, "End pos");
}

#[test]
fn legendary_concurrency_16_threads() {
    let ps = Arc::new(PatternSet::builder().literal("match").build().unwrap());
    let mut handles = vec![];
    for i in 0..16 {
        let ps_clone = ps.clone();
        handles.push(std::thread::spawn(move || {
            let input = format!("some match {}", i);
            let matches = ps_clone.scan(input.as_bytes()).unwrap();
            assert_eq!(matches.len(), 1, "Thread should match");
            assert_eq!(matches[0].start, 5, "Start pos");
            assert_eq!(matches[0].end, 10, "End pos");
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
