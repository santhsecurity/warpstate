# warpstate

<p align="center">
<a href="https://crates.io/crates/warpstate"><img src="https://img.shields.io/crates/v/warpstate.svg" alt="crates.io"></a>
<a href="https://docs.rs/warpstate"><img src="https://img.shields.io/docsrs/warpstate" alt="docs.rs"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
</p>

GPU-accelerated multi-pattern matching — run thousands of literals on GPU via wgpu compute shaders. Falls back to SIMD-accelerated Aho-Corasick on CPU when GPU is unavailable.

## Installation

```bash
cargo add warpstate
```

## Quick Example

```rust
use warpstate::{Match, PatternSet};

let patterns = PatternSet::builder()
    .literal("password")
    .literal("secret_key")
    .build()
    .unwrap();

// CPU scan — O(n) SIMD-accelerated Aho-Corasick
let matches = patterns.scan(b"the password is secret_key").unwrap();
assert_eq!(matches.len(), 2);
```

## Architecture Overview

```
PatternSet (builder API)
    → RegexDFA (compiled: dense transition matrix)
    → AhoCorasick (SIMD-accelerated)
        → CPU path: O(n) single-pass Aho-Corasick
        → GPU path: wgpu compute shader (one thread per byte)
    → AutoMatcher (routes by input size)
    → Batch API (coalesces many items → single GPU dispatch)
```

- **Zero panics**: Every operation returns `Result`
- **`#![forbid(unsafe_code)]`**: Pure safe Rust
- **GPU/CPU parity**: Same transition matrix on both paths

## Extension Guide

### Add a Custom Matcher

```rust
use warpstate::Matcher;

struct MyMatcher;

impl Matcher for MyMatcher {
    fn scan(&self, input: &[u8]) -> Result<Vec<Match>, Error> {
        // Your matching logic
    }
}
```

### Platform Backends

New GPU backends can be added by implementing the `GpuBackend` trait:

```rust
pub trait GpuBackend {
    async fn compile(&self, dfa: &Dfa) -> Result<Shader, Error>;
    async fn dispatch(&self, shader: &Shader, input: &[u8]) -> Result<Vec<Match>, Error>;
}
```

Contributions welcome for CUDA, Metal, and Vulkan-specific backends.

## License

MIT. Copyright 2026 Corum Collective LLC.
