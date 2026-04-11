#![warn(missing_docs)]
#![forbid(unsafe_code)]
//! # warpstate — GPU-accelerated multi-pattern matching
//!
//! Run thousands of literal patterns against large data using GPU compute
//! shaders via [wgpu](https://wgpu.rs). Falls back to CPU (Aho-Corasick)
//! when no GPU is available or when input is small enough that CPU is faster.
//!
//! # Quick Start
//!
//! ```rust
//! use warpstate::{PatternSet, Match};
//!
//! let patterns = PatternSet::builder()
//!     .literal("password")
//!     .literal("secret_key")
//!     .build()
//!     .unwrap();
//!
//! // CPU scan (always available, fast for small inputs)
//! let matches = patterns.scan(b"the password is secret_key").unwrap();
//! assert_eq!(matches.len(), 2);
//! ```
//!
//! # GPU Scan
//!
//! ```rust,no_run
//! # async fn example() {
//! # #[cfg(feature = "gpu")] {
//! use warpstate::{PatternSet, GpuMatcher};
//!
//! let patterns = PatternSet::builder()
//!     .literal("password")
//!     .literal("secret_key")
//!     .build()
//!     .unwrap();
//!
//! // GPU scan (10-100x faster on large inputs with many patterns)
//! let gpu = GpuMatcher::new(&patterns).await.unwrap();
//! let matches = gpu.scan(b"the password is secret_key").await.unwrap();
//! # }
//! # }
//! ```
//!
//! # Auto-Routing
//!
//! ```rust,no_run
//! # async fn example() {
//! use warpstate::{PatternSet, AutoMatcher};
//!
//! let patterns = PatternSet::builder()
//!     .literal("password")
//!     .build()
//!     .unwrap();
//!
//! let matcher = AutoMatcher::new(&patterns).await.unwrap();
//! // Automatically picks CPU for small inputs, GPU for large ones
//! let matches = matcher.scan(b"data...").await.unwrap();
//! # }
//! ```
//!
//! # Architecture
//!
//! ```text
//! PatternSet (builder API)
//!     → PatternIR (compiled: packed bytes + offset table)
//!         → CpuBackend (Aho-Corasick — always available)
//!         → GpuBackend (wgpu compute — when GPU present)
//!     → AutoMatcher (routes by input size)
//! ```
//!
//! The IR compiles once and is consumed by any backend. Adding a new backend
//! means implementing one function that takes `&PatternIR` and `&[u8]`.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::todo,
        clippy::unimplemented,
        clippy::expect_used,
        clippy::panic
    )
)]
// Performance-critical matching engine: pedantic lints are reviewed and selectively allowed.
#![allow(
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::unnecessary_literal_bound,
    clippy::unnecessary_wraps,
    clippy::needless_raw_string_hashes,
    clippy::unused_async,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::inline_always,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::explicit_counter_loop,
    clippy::match_same_arms,
    clippy::module_name_repetitions,
    clippy::manual_let_else,
    clippy::manual_slice_size_calculation,
    clippy::unused_self,
    clippy::needless_pass_by_value,
    clippy::redundant_closure_for_method_calls,
    clippy::manual_memcpy,
    dead_code
)]

pub mod batch;
pub mod compiled_index;
pub(crate) mod config;
pub(crate) mod cpu;
pub(crate) mod dfa;
pub(crate) mod error;
#[cfg(feature = "fused")]
pub(crate) mod fused;
#[cfg(feature = "gpu")]
pub mod gpu;
pub(crate) mod hash_scan;
mod literal_prefilter;
pub(crate) mod pattern;
pub(crate) mod regex_engine;
pub(crate) mod router;
pub(crate) mod scanner;
pub(crate) mod specialize;

#[cfg(feature = "gpu")]
pub(crate) mod dma;
/// Trait definition for pattern matching backends.
pub(crate) mod matcher;
#[cfg(feature = "gpu")]
pub(crate) mod multi_gpu;
pub(crate) mod pipeline;
pub(crate) mod stream;

pub use compiled_index::CompiledPatternIndex;
pub use config::{AutoMatcherConfig, DEFAULT_GPU_THRESHOLD, DEFAULT_MAX_MATCHES};
pub use cpu::{scan, scan_aho_corasick, scan_count, scan_with, CachedScanner};
pub use dfa::RegexDFA;
pub use error::{Error, Result};
#[cfg(feature = "fused")]
pub use fused::FusedScanner;
#[cfg(feature = "gpu")]
pub use gpu::GpuMatcher;
#[cfg(feature = "gpu")]
pub use gpu::GpuScanner;
pub use hash_scan::HashScanner;
pub use pattern::{HotSwapPatternSet, PatternIR, PatternSet, PatternSetBuilder};
pub use router::AutoMatcher;
pub use scanner::ByteScanner;
pub use specialize::ScanStrategy;

// Re-export vocabulary types from matchkit so existing consumers
// can continue using `warpstate::Match`, `warpstate::Matcher`, etc.
pub use matchkit::{BlockMatcher, BoxedMatcher, Matcher};
pub use matchkit::{GpuMatch, Match};

// Exports from newly wired modules
#[cfg(feature = "gpu")]
pub use multi_gpu::MultiGpuMatcher;
pub use pipeline::StreamPipeline;
pub use stream::StreamScanner;

// Match, GpuMatch, Matcher, BlockMatcher, and BoxedMatcher are now
// defined in the `matchkit` vocabulary crate and re-exported above.
// The `ZeroCopyBlockMatcher` trait stays here because it references
// GPU-specific `DmaStagingBuffer`.
