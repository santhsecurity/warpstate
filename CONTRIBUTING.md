# Contributing to warpstate

Thank you for your interest! warpstate is built for community extension.

## Pick your adventure

| I want to... | Read this | Difficulty |
|---|---|---|
| Add a backend (SIMD, Metal, Vulkan, etc.) | [`backends/README.md`](backends/README.md) | 🟡 Medium |
| Add a pattern type (regex, wildcards, case-insensitive) | `core/src/pattern.rs` | 🔴 Hard |
| Add a language binding (Python, Node.js, C, etc.) | [`bindings/`](bindings/) | 🟡 Medium |
| Optimize the GPU shader | `core/src/shader.rs` | 🔴 Hard |
| Improve the CPU baseline | `core/src/cpu.rs` | 🟢 Easy |

## Quick start

```bash
git clone https://github.com/santhsecurity/warpstate
cd warpstate

# Run tests (GPU tests skip automatically if no GPU)
cargo test -p warpstate-core

# Run CPU benchmarks
cargo run -p warpstate-core --release --example quick_bench

# Run clippy (must pass at pedantic level)
cargo clippy -p warpstate-core
```

## Code standards

- **`#![forbid(unsafe_code)]`** — no unsafe, period
- **Zero `unwrap()` or `expect()` outside tests**
- **All public items must have doc comments**
- **`cargo clippy` at pedantic level must pass with zero warnings**
- **Every PR must include tests and benchmarks**

## Good first issues

1. **Add case-insensitive matching** — compile-time case folding in `PatternIR`
2. **Add wildcard patterns** — `foo*bar` support in the pattern compiler
3. **Add SIMD backend** — Aho-Corasick with AVX2/AVX-512 intrinsics
4. **Improve GPU shader** — shared memory tiling for better cache utilization
5. **Add Python bindings** — PyO3 wrapper around `PatternSet` and `scan()`

## License

By contributing, you agree your contributions will be licensed under MIT.
