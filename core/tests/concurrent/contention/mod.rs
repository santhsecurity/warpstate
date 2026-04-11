#![cfg(feature = "gpu")]
use std::sync::Arc;
use std::thread;
use warpstate::*;

#[test]
fn test_gpu_resource_pool_contention() {
    let ps = Arc::new(PatternSet::builder().literal("GPU_TEST").build().unwrap());

    // Fall back to CPU if no GPU available to not fail the test
    let has_gpu = match pollster::block_on(AutoMatcher::new(&*ps)) {
        Ok(m) => m.has_gpu(),
        Err(_) => false,
    };

    if !has_gpu {
        return; // Nothing to test regarding GPU resource pool contention if there is no GPU
    }

    let mut handles = vec![];

    // If GPU threshold is small, we need an input bigger than it to hit GPU.
    // warpstate uses a 64KB threshold for GPU routing by default
    const GPU_THRESHOLD: usize = 65_536; // 64KB from router.rs
    let input_size = GPU_THRESHOLD + 10_000;

    for _i in 0..10 {
        let ps_clone = Arc::clone(&ps);
        handles.push(thread::spawn(move || {
            let mut data = vec![b'A'; input_size];
            data[50_000..50_008].copy_from_slice(b"GPU_TEST");

            // Should route through the consolidated GPU backend when AutoMatcher selects GPU.
            // For now use AutoMatcher
            let matcher = pollster::block_on(AutoMatcher::new(&*ps_clone)).unwrap();
            let matches = pollster::block_on(matcher.scan(&data)).unwrap();

            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].start, 50_000);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
