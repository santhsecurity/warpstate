# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive documentation for all public APIs.
- `GpuBufferPool` with size class documentation and poison recovery notes.
- Adversarial test suite covering buffer pool, config, and pattern compilation.
- Criterion benchmarks for config construction, pattern compilation, and buffer pool operations.
- CI workflow for automated testing, linting, and documentation builds.
- GPU device loss handling documentation.
- Thread safety tests for `GpuMatcher` and `GpuBufferPool`.

### Changed

- Improved module-level documentation for all source files.
- Enhanced `README.md` with architecture details and GPU requirements.

### Fixed

- Documentation links in public APIs.

## [0.1.0] - 2026-03-29

### Added

- Initial public release of warpstate.
- GPU-accelerated pattern matching via wgpu compute shaders.
- CPU fallback using SIMD-accelerated Aho-Corasick.
- Support for literal and regex patterns.
- Auto-routing between CPU and GPU backends.
- Batch scan API for processing multiple items.
- Double-buffered persistent GPU matcher.
- Algebraic parallel DFA execution via prefix scan.
- SMEM-staged DFA matching for small pattern sets.
- Rolling-hash GPU matcher for exact byte sequences.
- Zero-copy DMA staging buffer integration.
