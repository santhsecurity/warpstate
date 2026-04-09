//! Loom Testing Documentation
//!
//! Warpstate utilizes shared mutable state during multiple concurrent scans on the same engine.
//! For example, `GpuMatcher`, `SmemMatcher`, and `AlgebraicDfaMatcher` employ an internal
//! `Mutex<Option<State>>` that is taken (via a yielding spin-lock) during a scan.
//!
//! # Why Loom is N/A
//! Loom is Not Applicable (N/A) for testing Warpstate's concurrency. The core synchronization
//! involves hardware-backed queues, tokio background threads, and C-FFI across `wgpu`
//! (Vulkan/Metal/DX12). Loom requires absolute control over the thread scheduler and *all*
//! synchronization primitives down to the lowest level. Because `wgpu` manages its own
//! internal thread pools, memory fences, and locks that cannot be swapped with `loom::sync`
//! or `loom::thread`, any `loom::model` attempting to simulate concurrent GPU dispatch
//! would panic or yield undefined behavior due to escaping the modeled scheduler.
//! Therefore, Loom is fundamentally incompatible and N/A.
