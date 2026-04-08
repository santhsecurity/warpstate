use std::time::Instant;
use warpstate::PatternSet;

fn main() {
    let keywords = [
        "password",
        "secret",
        "api_key",
        "token",
        "authorization",
        "private_key",
        "access_key",
        "client_secret",
        "database_url",
        "aws_secret",
        "ssh-rsa",
        "BEGIN RSA PRIVATE KEY",
        "ghp_",
        "sk-",
        "Bearer ",
        "Basic ",
        "AKIA",
        "mysql://",
        "postgres://",
        "mongodb://",
    ];

    let mut builder = PatternSet::builder();
    for kw in &keywords {
        builder = builder.literal(*kw);
    }
    let patterns = builder.build().unwrap();

    // Generate test data
    let data: Vec<u8> = (0..10_000_000)
        .map(|i| b"abcdefghijklmnopqrstuvwxyz "[i % 27])
        .collect();

    println!("warpstate CPU benchmark");
    println!("======================");
    println!("Input: {} MB", data.len() / 1_048_576);
    println!("Patterns: {}", keywords.len());
    println!();

    // Warm up
    let _ = patterns.scan(&data[..1000]).unwrap();

    // Benchmark different sizes
    for size_mb in [1, 5, 10] {
        let size = size_mb * 1_048_576;
        let slice = &data[..size.min(data.len())];

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = patterns.scan(slice).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iterations;
        let throughput = (slice.len() as f64 / per_iter.as_secs_f64()) / 1_048_576.0;

        println!(
            "  {}MB: {:.2}ms/scan, {:.0} MB/s throughput",
            size_mb,
            per_iter.as_secs_f64() * 1000.0,
            throughput,
        );
    }

    // Many patterns benchmark
    println!();
    println!("Pattern scaling (1MB input):");
    let data_1mb = &data[..1_048_576];
    for count in [10, 100, 500, 1000, 5000] {
        let mut builder = PatternSet::builder();
        for i in 0..count {
            builder = builder.literal(&format!("pat{i:04}"));
        }
        let ps = builder.build().unwrap();

        let iterations = 5;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ps.scan(data_1mb).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iterations;
        let throughput = (data_1mb.len() as f64 / per_iter.as_secs_f64()) / 1_048_576.0;

        println!(
            "  {} patterns: {:.2}ms/scan, {:.0} MB/s",
            count,
            per_iter.as_secs_f64() * 1000.0,
            throughput,
        );
    }
}
