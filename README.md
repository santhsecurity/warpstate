<p align="center">
<h1 align="center">warpstate</h1>
<p align="center"><strong>GPU-accelerated multi-pattern matching</strong></p>
<p align="center">
<a href="https://crates.io/crates/warpstate"><img src="https://img.shields.io/crates/v/warpstate.svg" alt="crates.io"></a>
<a href="https://docs.rs/warpstate"><img src="https://img.shields.io/docsrs/warpstate" alt="docs.rs"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
</p>
</p>

---

Run thousands of patterns against large data using GPU compute shaders via [wgpu](https://wgpu.rs). Falls back to SIMD-accelerated Aho-Corasick on CPU when no GPU is available or when input is small enough that CPU is faster.

> *Most pattern matchers are CPU-bound. warpstate compiles your patterns into a
> dense DFA transition matrix and executes it as a WGSL compute shader — one
> GPU thread per byte position. On large inputs with many patterns, this is
> 10-100x faster than sequential CPU scanning.*

## Quick Start

```rust
use warpstate::{Match, PatternSet};

let patterns = PatternSet::builder()
    .literal("password")
    .literal("secret_key")
    .build()
    .unwrap();

// CPU scan — O(n) SIMD-accelerated Aho-Corasick for literal patterns
let matches = patterns.scan(b"the password is secret_key").unwrap();
assert_eq!(matches.len(), 2);
```

## GPU Scan

```rust,no_run
# async fn example() {
use warpstate::{PatternSet, WarpStateer};

let patterns = PatternSet::builder()
    .literal("password")
    .literal("secret_key")
    .build()
    .unwrap();

// GPU scan (10-100x faster on large inputs with many patterns)
let gpu = WarpStateer::new(&patterns).await.unwrap();
let matches = gpu.scan(b"the password is secret_key").await.unwrap();
# }
```

## Auto-Routing

```rust,no_run
# async fn example() {
use warpstate::{AutoMatcher, PatternSet};

let patterns = PatternSet::builder()
    .literal("password")
    .build()
    .unwrap();

let matcher = AutoMatcher::new(&patterns).await.unwrap();
// Automatically picks CPU for small inputs, GPU for large ones
let matches = matcher.scan(b"data...").await.unwrap();
# }
```

## Batch Scan — The Killer Feature

Scan thousands of small files in a single GPU dispatch:

```rust
use warpstate::{PatternSet, ScanItem};

let patterns = PatternSet::builder()
    .literal("secret")
    .literal("password")
    .build()
    .unwrap();

let items = vec![
    ScanItem { id: 0, data: b"clean file" },
    ScanItem { id: 1, data: b"has a secret" },
    ScanItem { id: 2, data: b"password leak" },
];

let tagged = patterns.scan_batch(items).unwrap();
// Each match is tagged with its source item ID
for t in &tagged {
    println!("item {} matched pattern {}", t.source_id, t.matched.pattern_id);
}
```

With `AutoMatcher::scan_batch()`, items are coalesced into a single contiguous
buffer and dispatched as one GPU compute pass — amortizing GPU overhead across
hundreds of small inputs. This makes GPU scanning viable even for 4KB files
when batched together.

## Architecture

```text
PatternSet (builder API)
    → RegexDFA (compiled: dense transition matrix + byte equivalence classes)
    → AhoCorasick (SIMD-accelerated, built for literal-only sets)
        → CPU path: O(n) single-pass Aho-Corasick
        → GPU path: wgpu compute shader (one thread per byte)
    → AutoMatcher (routes by input size, auto-tunes threshold)
    → Batch API (coalesces many items → single GPU dispatch)
```

| Layer | Purpose |
|---|---|
| `PatternSet` | Builder API, compiles patterns into both DFA and Aho-Corasick |
| `RegexDFA` | Dense transition matrix with byte equivalence classes for GPU |
| `AhoCorasick` | SIMD-accelerated literal matching for CPU (via BurntSushi's crate) |
| `WarpStateer` | GPU compute pipeline — uploads DFA, dispatches shader, reads matches |
| `StreamPipeline` | Chunks arbitrarily large inputs for GPU's fixed buffer size |
| `AutoMatcher` | Routes between CPU/GPU based on input size, auto-tunes threshold |
| `batch` | Coalesces multiple items for amortized GPU dispatch |

## CPU Algorithm Selection

| Pattern types | Algorithm | Complexity |
|---|---|---|
| All literals | Aho-Corasick (SIMD) | O(n) |
| Mixed/regex | DFA walk | O(n × depth) |
| GPU | Dense DFA shader | O(n / threads) |

## Engineering Standards

- **Zero panics**: Every operation returns `Result`. Zero `unwrap()` in library code.
- **`#![forbid(unsafe_code)]`**: The entire core crate is safe Rust.
- **Clippy pedantic**: Zero warnings at `clippy::pedantic` level.
- **GPU/CPU parity**: `scan_dfa()` walks the exact same transition matrix as the GPU shader for correctness testing.
- **Extensive tests**: Unit, adversarial, property-based, and fuzz coverage for chunking and pipeline boundaries.

## Feature Flags

| Flag | Default | Purpose |
|---|---|---|
| *(none yet)* | | All features are included by default |

## License

MIT. Copyright 2026 CORUM COLLECTIVE LLC.
