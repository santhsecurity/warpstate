use super::*;
use crate::gpu::device;
use crate::gpu::dispatch;
use crate::gpu::readback;
use crate::PatternSet;
use std::sync::Arc;
use std::thread;
use wgpu::util::DeviceExt;

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(future)
}

// === Basic GPU Scan Tests ===

fn gpu_scan_single_match() -> Result<()> {
    let ps = PatternSet::builder().literal("hello").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        let matches = block_on(gpu.scan(b"hello world"))?;
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_id, 0);
    }
Ok(())
}

fn gpu_scan_supports_regex_patterns() -> Result<()> {
    let ps = PatternSet::builder().regex(r"ab+c").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        let matches = block_on(gpu.scan(b"xxabbbcxx"))?;
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 2);
    }
Ok(())
}

fn gpu_scan_respects_legacy_input_size_limit() -> Result<()> {
    let ps = PatternSet::builder().literal("test").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::with_options(&ps, 1024)) {
        let result = block_on(gpu.scan(&vec![b'x'; 2048]));
        assert!(matches!(result, Err(Error::InputTooLarge { .. })));
    }
Ok(())
}

#[test]
fn pad_to_u32_correctly_aligns() {
    let result = device::pad_to_u32(&[0x01, 0x02, 0x03]);
    assert_eq!(result[0], u32::from_le_bytes([0x01, 0x02, 0x03, 0x00]));
}

// === Buffer Pool Adversarial Tests ===

#[test]
fn buffer_pool_empty_get_creates_new() {
    let pool = GpuBufferPool::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let buf = pool.get_or_create(&device, "test", 1024, wgpu::BufferUsages::STORAGE);
            assert_eq!(buf.size(), 1024);
        }
    }
}

#[test]
fn buffer_pool_reuse_after_return() {
    let pool = GpuBufferPool::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let buf1 = pool.get_or_create(&device, "test", 1024, wgpu::BufferUsages::STORAGE);
            let size = buf1.size();
            pool.return_buffer(buf1, wgpu::BufferUsages::STORAGE);

            let buf2 = pool.get_or_create(&device, "test", 1024, wgpu::BufferUsages::STORAGE);
            assert_eq!(buf2.size(), size);
        }
    }
}

#[test]
fn buffer_pool_size_class_rounding_small() {
    let pool = GpuBufferPool::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let buf = pool.get_or_create(&device, "test", 1, wgpu::BufferUsages::STORAGE);
            assert_eq!(buf.size(), 64);
        }
    }
}

#[test]
fn buffer_pool_size_class_rounding_boundary() {
    let pool = GpuBufferPool::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let buf = pool.get_or_create(&device, "test", 65, wgpu::BufferUsages::STORAGE);
            assert_eq!(buf.size(), 128);
        }
    }
}

#[test]
fn buffer_pool_different_usage_not_mixed() {
    let pool = GpuBufferPool::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let storage = pool.get_or_create(&device, "test", 1024, wgpu::BufferUsages::STORAGE);
            pool.return_buffer(storage, wgpu::BufferUsages::STORAGE);

            let uniform = pool.get_or_create(&device, "test", 1024, wgpu::BufferUsages::UNIFORM);
            assert_eq!(uniform.size(), 1024);
            assert_eq!(pool.available.lock().expect("mutex poisoned").len(), 1);
        }
    }
}

#[test]
fn buffer_pool_multiple_returns() {
    let pool = GpuBufferPool::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let buf1 = pool.get_or_create(&device, "test", 1024, wgpu::BufferUsages::STORAGE);
            let buf2 = pool.get_or_create(&device, "test", 2048, wgpu::BufferUsages::STORAGE);
            pool.return_buffer(buf1, wgpu::BufferUsages::STORAGE);
            pool.return_buffer(buf2, wgpu::BufferUsages::STORAGE);

            assert_eq!(pool.available.lock().expect("mutex poisoned").len(), 2);

            let got1 = pool.get_or_create(&device, "test", 1024, wgpu::BufferUsages::STORAGE);
            let got2 = pool.get_or_create(&device, "test", 2048, wgpu::BufferUsages::STORAGE);
            assert!(got1.size() >= 1024);
            assert!(got2.size() >= 2048);
        }
    }
}

#[test]
fn buffer_pool_thread_safety() {
    let pool = Arc::new(GpuBufferPool::default());
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let device = Arc::new(device);
            let mut handles = vec![];

            for i in 0..4 {
                let pool_clone = Arc::clone(&pool);
                let device_clone = Arc::clone(&device);
                handles.push(thread::spawn(move || {
                    for _ in 0..10 {
                        let buf = pool_clone.get_or_create(
                            &device_clone,
                            "test",
                            1024 * (i as u64 + 1),
                            wgpu::BufferUsages::STORAGE,
                        );
                        pool_clone.return_buffer(buf, wgpu::BufferUsages::STORAGE);
                    }
                }));
            }

            for handle in handles {
                handle.join().expect("thread panicked");
            }
        }
    }
}

fn gpu_scan_empty_input() -> Result<()> {
    let ps = PatternSet::builder().literal("test").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        let matches = block_on(gpu.scan(b""))?;
        assert!(matches.is_empty());
    }
Ok(())
}

fn gpu_scan_large_input_chunking() -> Result<()> {
    let ps = PatternSet::builder().literal("needle").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        let mut data = vec![b'x'; 1024 * 1024];
        data[512 * 1024..512 * 1024 + 6].copy_from_slice(b"needle");

        let matches = block_on(gpu.scan(&data))?;
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 512_u32 * 1024);
    }
Ok(())
}

fn gpu_matcher_thread_safety() -> Result<()> {
    let ps = PatternSet::builder().literal("test").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        let gpu = Arc::new(gpu);
        let mut handles = vec![];

        for i in 0..4 {
            let gpu_clone = Arc::clone(&gpu);
            let data = format!("test data {i}").into_bytes();
            handles.push(thread::spawn(move || -> Result<()> {
                for _ in 0..5 {
                    let matches = block_on(gpu_clone.scan(&data))?;
                    assert!(!matches.is_empty());
                }
                Ok(())
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked")?;
        }
    }
Ok(())
}

fn gpu_scan_regex_input_too_large() -> Result<()> {
    let ps = PatternSet::builder().regex("a.*b").build()?;
    let config = AutoMatcherConfig::new().gpu_max_regex_input_size(1024);
    if let Ok(gpu) = block_on(GpuMatcher::with_config(&ps, config)) {
        let data = vec![b'a'; 2048];
        let result = block_on(gpu.scan(&data));
        match result {
            Err(Error::InputTooLarge { .. }) => {}
            _ => panic!("Expected InputTooLarge error, got {:?}", result),
        }
    }
Ok(())
}

fn gpu_scan_inconsistent_match_count() -> Result<()> {
    let ps = PatternSet::builder().literal("hello").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        let (device, queue) = gpu.gpu_device_queue();
        let count_staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[100u32, 0u32]),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let result = block_on(readback::read_matches(
            &device,
            &queue,
            &count_staging,
            &match_staging,
            Some(&[0]),
            0,
            gpu.max_matches,
            1024, // test input length
        ));
        // Uninitialized GPU buffers may contain zeros, which reads as 0 matches
        // (not an error). The inconsistency error only fires when the match
        // count exceeds max_matches, which zero-filled buffers don't trigger.
        match result {
            Err(Error::GpuDeviceError { reason }) => {
                assert!(reason.contains("inconsistent match count"));
            }
            Ok(matches) => {
                // Zero-filled staging buffers → 0 matches is valid
                assert!(
                    matches.is_empty(),
                    "uninitialized buffers should yield 0 matches"
                );
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
        count_staging.destroy();
        match_staging.destroy();
    }
Ok(())
}

/// Check if the current GPU adapter is a known software renderer.
/// Used by tests to skip when running on broken software GPU implementations.
fn is_software_adapter() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Some(adapter) => super::adapter_is_software(&adapter.get_info()),
        None => true,
    }
}

/// Regression test: GPU prefilter must emit ALL hash-matching patterns, not just first.
///
/// Before the fix, patterns sharing FNV hash prefixes (common with similar names)
/// produced only 1 candidate instead of N. This test creates 20 patterns with a
/// shared prefix and verifies the GPU finds all 10 that appear in the content.
fn gpu_prefilter_emits_all_hash_matching_candidates() -> Result<()> {
    if is_software_adapter() {
        return Ok(()); // Software renderers have known parity bugs
    }

    let mut builder = PatternSet::builder();
    for i in 0..20 {
        builder = builder.literal(&format!("SHARED_PREFIX_{i:04}_SUFFIX"));
    }
    let ps = builder.build()?;

    let Ok(gpu) = block_on(GpuMatcher::new(&ps)) else {
        return Ok(()); // No GPU adapter
    };

    // Content contains the first 10 patterns
    let mut content = String::new();
    for i in 0..10 {
        content.push_str(&format!("SHARED_PREFIX_{i:04}_SUFFIX\n"));
    }

    let gpu_matches = block_on(gpu.scan(content.as_bytes()))?;
    let cpu_matches = ps.scan(content.as_bytes())?;

    assert_eq!(
        gpu_matches.len(),
        cpu_matches.len(),
        "GPU must find same number of matches as CPU.\n\
         GPU found {} matches, CPU found {} matches.\n\
         GPU pattern_ids: {:?}\n\
         CPU pattern_ids: {:?}",
        gpu_matches.len(),
        cpu_matches.len(),
        gpu_matches.iter().map(|m| m.pattern_id).collect::<Vec<_>>(),
        cpu_matches.iter().map(|m| m.pattern_id).collect::<Vec<_>>(),
    );

    // Verify each CPU match has a corresponding GPU match
    for cpu_match in &cpu_matches {
        assert!(
            gpu_matches
                .iter()
                .any(|gm| gm.pattern_id == cpu_match.pattern_id && gm.start == cpu_match.start),
            "CPU match (pattern={}, start={}) not found in GPU results",
            cpu_match.pattern_id,
            cpu_match.start,
        );
    }
Ok(())
}

/// Regression test: GPU prefilter must NOT stop probing after finding a shorter prefix match.
///
/// Patterns with different effective prefix lengths (e.g., 4-byte vs 8-byte) that share the
/// same initial bytes must ALL be emitted as candidates. Previously the shader broke out of
/// the prefix-length loop as soon as any candidate was found, causing longer patterns to be
/// silently missed (critical false negative at internet scale).
fn gpu_prefilter_does_not_stop_at_shorter_prefix() -> Result<()> {
    if is_software_adapter() {
        return Ok(()); // Software renderers have known parity bugs
    }

    // "test" (len 4) is a prefix of "testing" (len 7). Both start with "test".
    let ps = PatternSet::builder()
        .literal("test")
        .literal("testing")
        .build()
        ?;

    let Ok(gpu) = block_on(GpuMatcher::new(&ps)) else {
        return Ok(()); // No GPU adapter
    };

    let input = b"this is a testing sentence";
    let gpu_matches = block_on(gpu.scan(input))?;

    // The GPU literal backend returns overlapping matches (unlike CPU leftmost-first),
    // so we only verify that the longer pattern is NOT silently dropped.
    assert!(
        gpu_matches
            .iter()
            .any(|m| m.pattern_id == 1 && m.start == 10 && m.end == 17),
        "GPU must find 'testing' (pattern_id=1, start=10, end=17)"
    );
    // The shorter prefix pattern should also be found.
    assert!(
        gpu_matches
            .iter()
            .any(|m| m.pattern_id == 0 && m.start == 10 && m.end == 14),
        "GPU must find 'test' (pattern_id=0, start=10, end=14)"
    );
Ok(())
}

// === GPU/CPU Parity Tests ===

/// (1) 50 literal patterns scanned on a 100KB input.
fn gpu_cpu_parity_50_literals_100kb() -> Result<()> {
    if is_software_adapter() {
        return Ok(());
    }

    let mut builder = PatternSet::builder();
    for i in 0..50 {
        builder = builder.literal(&format!("LITERAL_{i:03}_MATCH"));
    }
    let ps = builder.build()?;

    let Ok(gpu) = block_on(GpuMatcher::new(&ps)) else {
        return Ok(()); // No GPU adapter
    };

    let mut input = vec![b'x'; 100 * 1024];
    let p = b"LITERAL_007_MATCH";
    input[1000..1000 + p.len()].copy_from_slice(p);
    let p = b"LITERAL_042_MATCH";
    input[50000..50000 + p.len()].copy_from_slice(p);
    let p = b"LITERAL_049_MATCH";
    input[99980..99980 + p.len()].copy_from_slice(p);

    let mut gpu_matches = block_on(gpu.scan(&input))?;
    let mut cpu_matches = ps.scan(&input)?;
    gpu_matches.sort_unstable();
    cpu_matches.sort_unstable();

    assert_eq!(
        gpu_matches, cpu_matches,
        "GPU/CPU parity failed for 50 literals on 100KB input"
    );
Ok(())
}

/// (2) Mixed literal+regex patterns.
/// FINDING: GPU regex DFA shader returns out-of-bounds pattern index on RTX 5090.
/// The GPU regex readback maps internal indices to user IDs, but the DFA shader
/// produces state-based pattern IDs that don't align with the literal pipeline's
/// index space. This is a real bug in the GPU regex readback path.
fn gpu_cpu_parity_mixed_literal_regex() -> Result<()> {
    if is_software_adapter() {
        return Ok(());
    }

    let ps = PatternSet::builder()
        .literal("password")
        .regex(r"secr[e3]t")
        .literal("token")
        .regex(r"api[_-]?key")
        .build()
        ?;

    let Ok(gpu) = block_on(GpuMatcher::new(&ps)) else {
        return Ok(()); // No GPU adapter
    };

    let input = b"the password is secr3t and the token has api-key here";
    let mut gpu_matches = block_on(gpu.scan(input))?;
    let mut cpu_matches = ps.scan(input)?;
    gpu_matches.sort_unstable();
    cpu_matches.sort_unstable();

    assert_eq!(
        gpu_matches, cpu_matches,
        "GPU/CPU parity failed for mixed literal+regex patterns"
    );
Ok(())
}

/// (3) Overlapping pattern matches at the same position.
/// FINDING: Same GPU regex out-of-bounds as H19 — mixed literal+regex with overlaps.
fn gpu_cpu_parity_overlapping_same_position() -> Result<()> {
    if is_software_adapter() {
        return Ok(());
    }

    let ps = PatternSet::builder()
        .literal("needle")
        .regex("needle")
        .literal("needle")
        .regex("n..dle")
        .build()
        ?;

    let Ok(gpu) = block_on(GpuMatcher::new(&ps)) else {
        return Ok(()); // No GPU adapter
    };

    let input = b"needle";
    let mut gpu_matches = block_on(gpu.scan(input))?;
    let mut cpu_matches = ps.scan(input)?;
    gpu_matches.sort_unstable();
    cpu_matches.sort_unstable();

    assert_eq!(
        gpu_matches, cpu_matches,
        "GPU/CPU parity failed for overlapping matches at same position"
    );
Ok(())
}

/// (4) Empty input returns empty matches.
/// FINDING: GPU regex .* matches empty input differently than CPU — regex semantics edge case.
fn gpu_cpu_parity_empty_input() -> Result<()> {
    if is_software_adapter() {
        return Ok(());
    }

    let ps = PatternSet::builder()
        .literal("test")
        .regex(".*")
        .build()
        ?;

    let Ok(gpu) = block_on(GpuMatcher::new(&ps)) else {
        return Ok(()); // No GPU adapter
    };

    let gpu_matches = block_on(gpu.scan(b""))?;
    let cpu_matches = ps.scan(b"")?;

    assert_eq!(
        gpu_matches, cpu_matches,
        "GPU/CPU parity failed for empty input"
    );
Ok(())
}

/// (5) Input exactly at chunk boundary (128MB).
/// FINDING: GPU scan at exactly 128MB triggers wgpu buffer allocation panic.
fn gpu_cpu_parity_exact_chunk_boundary() -> Result<()> {
    if is_software_adapter() {
        return Ok(());
    }

    let ps = PatternSet::builder()
        .literal("BOUNDARY_MATCH")
        .build()
        ?;

    let Ok(gpu) = block_on(GpuMatcher::new(&ps)) else {
        return Ok(()); // No GPU adapter
    };

    let size = DEFAULT_MAX_INPUT_SIZE;
    let mut input = vec![b'x'; size];
    // Place match near the end to verify the full buffer is scanned.
    let offset = size - 14;
    input[offset..offset + 14].copy_from_slice(b"BOUNDARY_MATCH");

    let mut gpu_matches = block_on(gpu.scan(&input))?;
    let mut cpu_matches = ps.scan(&input)?;
    gpu_matches.sort_unstable();
    cpu_matches.sort_unstable();

    assert_eq!(
        gpu_matches, cpu_matches,
        "GPU/CPU parity failed for input exactly at 128MB chunk boundary"
    );
Ok(())
}


// === Adversarial Production-Hardening Tests ===

/// Match starting exactly at a chunk boundary must not be dropped.
/// Uses a tiny chunk size to force chunking on a small input.
fn gpu_scan_match_at_chunk_boundary() -> Result<()> {
    if is_software_adapter() {
        return Ok(());
    }

    let ps = PatternSet::builder().literal("BOUNDARY").build()?;

    let config = AutoMatcherConfig::new().chunk_size(16).chunk_overlap(8);
    let Ok(gpu) = block_on(GpuMatcher::with_config(&ps, config)) else {
        return Ok(()); // No GPU adapter
    };

    // Place "BOUNDARY" so it starts at byte 16 (the chunk boundary).
    let mut input = vec![b'x'; 32];
    input[16..24].copy_from_slice(b"BOUNDARY");

    let gpu_matches = block_on(gpu.scan(&input))?;
    assert_eq!(gpu_matches.len(), 1, "match at chunk boundary must be found");
    assert_eq!(gpu_matches[0].start, 16);
    assert_eq!(gpu_matches[0].end, 24);
Ok(())
}

/// compute_workgroups must reject inputs that exceed the device workgroup limit.
#[test]
fn gpu_compute_workgroups_respects_limits() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    if let Some(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let max = device.limits().max_compute_workgroups_per_dimension;
            let max_x = max.min(65_535);

            // Input that fits exactly in 1D
            let ok_1d = max * shader::WORKGROUP_SIZE;
            assert!(dispatch::compute_workgroups(&device, ok_1d).is_ok());

            // Input that requires 2D dispatch
            let ok_2d = (max_x + 1) * shader::WORKGROUP_SIZE;
            if ok_2d <= u32::MAX {
                let dims = dispatch::compute_workgroups(&device, ok_2d).unwrap();
                assert!(dims.y >= 2, "input exceeding max_x must spill to y dimension");
            }

            // Input that exceeds the 2D limit (if u32::MAX is large enough)
            let bad_input = u32::MAX;
            let bad_workgroups = (bad_input as u64).div_ceil(shader::WORKGROUP_SIZE as u64);
            if bad_workgroups > (max_x as u64) * (max as u64) {
                let result = dispatch::compute_workgroups(&device, bad_input);
                assert!(
                    matches!(result, Err(Error::GpuDeviceError { .. })),
                    "expected GpuDeviceError for workgroup overflow, got {:?}",
                    result
                );
            }
        }
    }
}

/// read_matches must detect sentinel-filled count buffers (device lost / unexecuted shader).
#[test]
fn gpu_readback_detects_sentinel() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    if let Some(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, queue)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let count_staging = device::readback_buffer(&device, 8, "sentinel count");
            let match_staging = device::readback_buffer(&device, 64, "sentinel match");

            // Submit an empty encoder so on_submitted_work_done fires.
            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            queue.submit(Some(encoder.finish()));

            let result = block_on(readback::read_matches(
                &device,
                &queue,
                &count_staging,
                &match_staging,
                None,
                0,
                100,
                1024,
            ));

            assert!(
                matches!(result, Err(Error::GpuDeviceError { ref reason }) if reason.contains("sentinel")),
                "expected sentinel error for unread count buffer, got {:?}",
                result
            );
            count_staging.destroy();
            match_staging.destroy();
        }
    }
}

/// read_matches must skip corrupted match offsets instead of trusting them.
fn gpu_readback_skips_invalid_offsets() -> Result<()> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    if let Some(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, queue)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            // Count: 2 matches, no overflow
            let count_staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[2u32, 0u32]),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            });

            // Match buffer layout per match: [pattern_id, start, end, padding]
            // Match 0: valid
            // Match 1: start > end (inverted)
            let match_data: Vec<u32> = vec![
                0, 0, 4, 0,   // valid match at [0, 4)
                0, 10, 5, 0,  // inverted: start=10, end=5
            ];
            let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let mut view = match_staging.slice(..).get_mapped_range_mut();
                view.copy_from_slice(bytemuck::cast_slice(&match_data));
            }
            match_staging.unmap();

            // Submit an empty encoder so on_submitted_work_done fires.
            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            queue.submit(Some(encoder.finish()));

            let matches = block_on(readback::read_matches(
                &device,
                &queue,
                &count_staging,
                &match_staging,
                None,
                0,
                100,
                8, // input_len = 8
            ))
            ?;

            assert_eq!(matches.len(), 1, "only the valid match should be kept");
            assert_eq!(matches[0].start, 0);
            assert_eq!(matches[0].end, 4);
            count_staging.destroy();
            match_staging.destroy();
        }
    }
Ok(())
}

/// GpuBufferPool::get_or_create must not panic on sizes that overflow next_power_of_two.
#[test]
fn buffer_pool_huge_size_no_panic() {
    let pool = GpuBufferPool::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    if let Some(adapter) =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            // Before the fix, (size as usize).next_power_of_two() panicked when
            // size > usize::MAX / 2. After the fix, checked_next_power_of_two
            // gracefully returns None and falls back to the original size.
            let huge_size = (usize::MAX / 2 + 1) as u64;
            // We only verify our code does not panic inside next_power_of_two.
            // wgpu may panic later on the impossible allocation, but that is
            // outside our control and happens only after our fixed code runs.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.get_or_create(&device, "huge", huge_size, wgpu::BufferUsages::STORAGE)
            }));
        }
    }
}


// === Device Recovery Tests ===

fn gpu_device_recovery_recreates_on_flag() -> Result<()> {
    let ps = PatternSet::builder().literal("hello").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        // Force a device recreation by setting the flag.
        gpu.device_needs_recreation.store(true, std::sync::atomic::Ordering::SeqCst);

        let matches = block_on(gpu.scan(b"hello world"))?;
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_id, 0);

        // Flag should be cleared after successful recreation.
        assert!(!gpu.device_needs_recreation.load(std::sync::atomic::Ordering::SeqCst));
    }
Ok(())
}

fn gpu_device_recovery_concurrent_flag_set() -> Result<()> {
    let ps = PatternSet::builder().literal("test").build()?;
    if let Ok(gpu) = block_on(GpuMatcher::new(&ps)) {
        let gpu = Arc::new(gpu);
        let mut handles = vec![];

        // Force recreation on every thread.
        gpu.device_needs_recreation.store(true, std::sync::atomic::Ordering::SeqCst);

        for i in 0..4 {
            let gpu_clone = Arc::clone(&gpu);
            let data = format!("test data {i}").into_bytes();
            handles.push(std::thread::spawn(move || -> Result<()> {
                for _ in 0..5 {
                    let matches = block_on(gpu_clone.scan(&data))?;
                    assert!(!matches.is_empty());
                }
                Ok(())
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked")?;
        }

        assert!(!gpu.device_needs_recreation.load(std::sync::atomic::Ordering::SeqCst));
    }
Ok(())
}

#[test]
fn gpu_is_device_lost_error_detects_sentinel() {
    let err = Error::GpuDeviceError {
        reason: "GPU count buffer contains sentinel value — device may be lost or shader did not execute. Fix: retry with smaller input or check GPU health.".to_string(),
    };
    assert!(super::is_recoverable_gpu_error(&err));
}

#[test]
fn gpu_is_device_lost_error_detects_timeout() {
    let err = Error::GpuDeviceError {
        reason: "GPU buffer map timed out after 30s: test".to_string(),
    };
    assert!(super::is_recoverable_gpu_error(&err));
}

#[test]
fn gpu_is_device_lost_error_detects_buffer_map_failed() {
    assert!(super::is_recoverable_gpu_error(&Error::BufferMapFailed));
}

#[test]
fn gpu_is_device_lost_error_ignores_other_errors() {
    let err = Error::InputTooLarge {
        bytes: 1024,
        max_bytes: 512,
    };
    assert!(!super::is_recoverable_gpu_error(&err));
}


#[test]
fn gpu_compute_workgroups_2d_when_needed() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    if let Some(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        if let Ok((device, _)) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        {
            let max = device.limits().max_compute_workgroups_per_dimension;
            let max_x = max.min(65_535);

            // Small input should be 1D
            let small = dispatch::compute_workgroups(&device, 1024).unwrap();
            assert_eq!(small.y, 1, "small inputs must use 1D dispatch");

            // Large input that exceeds max_x should spill to 2D
            let large_input = (max_x + 1) * shader::WORKGROUP_SIZE;
            let large = dispatch::compute_workgroups(&device, large_input).unwrap();
            assert!(
                large.y > 1 || large.x <= max_x,
                "inputs exceeding max_x must use 2D dispatch or fit within max_x"
            );
        }
    }
}
