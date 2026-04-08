use std::sync::Arc;
use std::thread;
use warpstate::*;

#[test]
fn test_100_threads_scanning_different_inputs() {
    let ps = Arc::new(
        PatternSet::builder()
            .literal("COMMON_TARGET")
            .build()
            .unwrap(),
    );

    let mut handles = vec![];

    for i in 0..100 {
        let ps_clone = Arc::clone(&ps);
        handles.push(thread::spawn(move || {
            // Generate a slightly different input per thread
            let mut data = format!("prefix_{} ", i).into_bytes();
            data.extend_from_slice(b"COMMON_TARGET");
            data.extend_from_slice(format!(" _suffix_{}", i).as_bytes());

            let matches = ps_clone.scan(&data).unwrap();

            assert_eq!(matches.len(), 1);
            // Verify match is correct
            assert_eq!(
                &data[matches[0].start as usize..matches[0].end as usize],
                b"COMMON_TARGET"
            );
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
