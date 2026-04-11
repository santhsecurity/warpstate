//! Shareable synchronous scanner that executes via the GPU matcher.

use std::sync::Arc;

use super::GpuMatcher;
use crate::scanner::ByteScanner;
use crate::{Match, PatternSet, Result};

/// Shared synchronous scanner for production pipelines.
///
/// Construction eagerly attempts GPU initialization and pipeline compilation.
/// If initialization fails, scans fail with a GPU error.
#[derive(Debug, Clone)]
pub struct GpuScanner {
    patterns: Arc<PatternSet>,
    backend: Arc<Backend>,
}

#[derive(Debug)]
enum Backend {
    Gpu(Arc<GpuMatcher>),
    InitializationFailed { reason: String },
}

impl GpuScanner {
    /// Create a scanner from a compiled [`PatternSet`].
    #[must_use]
    pub fn new(patterns: PatternSet) -> Self {
        Self::from_arc(Arc::new(patterns))
    }

    /// Create a scanner from a shared [`PatternSet`].
    #[must_use]
    pub fn from_arc(patterns: Arc<PatternSet>) -> Self {
        let backend = match create_gpu_backend(patterns.as_ref()) {
            Ok(gpu) => Arc::new(Backend::Gpu(Arc::new(gpu))),
            Err(error) => Arc::new(Backend::InitializationFailed {
                reason: error.to_string(),
            }),
        };

        Self { patterns, backend }
    }

    /// Scan `data` using the GPU matcher.
    #[allow(clippy::needless_return)] // Returns needed for #[cfg(feature)] branching
    pub fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        match self.backend.as_ref() {
            Backend::Gpu(gpu) => {
                return gpu.scan_blocking(data);
            }
            Backend::InitializationFailed { reason } => {
                return Err(crate::Error::GpuDeviceError {
                    reason: format!(
                        "GPU scanner initialization failed: {reason}. \
                         Fix: verify GPU drivers are installed and wgpu can access the adapter. \
                         Use PatternSet::scan() for CPU-only scanning."
                    ),
                });
            }
        }
    }

    /// Scan `data` using the GPU backend.
    pub fn scan_gpu(&self, data: &[u8]) -> Result<Vec<Match>> {
        self.scan(data)
    }

    /// True when GPU initialization succeeded and scans will dispatch to wgpu.
    #[must_use]
    pub fn uses_gpu(&self) -> bool {
        matches!(self.backend.as_ref(), Backend::Gpu(_))
    }

    /// Scan using GPU if available, otherwise fall back to CPU.
    ///
    /// Unlike `scan()` which errors on GPU init failure, this method gracefully
    /// falls back to CPU scanning with a tracing warning. Use this in pipelines
    /// where GPU acceleration is preferred but not required.
    #[allow(clippy::needless_return)]
    pub fn scan_or_cpu_fallback(&self, data: &[u8]) -> Result<Vec<Match>> {
        match self.scan(data) {
            Ok(matches) => Ok(matches),
            Err(crate::Error::GpuDeviceError { reason }) => {
                tracing::warn!("GPU unavailable ({reason}), using CPU fallback");
                self.patterns.scan(data)
            }
            Err(other) => Err(other),
        }
    }

    /// Returns the GPU initialization failure if the scanner cannot be brought online.
    #[must_use]
    #[allow(clippy::needless_return)]
    pub fn fallback_reason(&self) -> Option<&str> {
        match self.backend.as_ref() {
            Backend::Gpu(_) => None,
            Backend::InitializationFailed { reason } => Some(reason.as_str()),
        }
    }
    /// Returns a reference to the underlying wgpu device and queue if GPU
    /// initialization succeeded.
    ///
    /// This allows crates that need the same GPU device (e.g., `gputokenize`
    /// for token-aware filtering) to share it without creating a new device.
    #[must_use]
    pub fn gpu_device_queue(&self) -> Option<(wgpu::Device, wgpu::Queue)> {
        match self.backend.as_ref() {
            Backend::Gpu(gpu) => Some(gpu.gpu_device_queue()),
            Backend::InitializationFailed { .. } => None,
        }
    }
}

impl ByteScanner for GpuScanner {
    fn scan_bytes(&self, data: &[u8]) -> Result<Vec<Match>> {
        self.scan_or_cpu_fallback(data)
    }
}

fn create_gpu_backend(patterns: &PatternSet) -> Result<GpuMatcher> {
    let patterns = patterns.clone();
    std::thread::spawn(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pollster::block_on(GpuMatcher::new(&patterns))))
            .map_err(|panic_payload| crate::Error::GpuDeviceError {
                reason: format!(
                    "GPU pipeline initialization panicked. Fix: validate the active GPU shaders or driver stack before enabling GPU scan mode. Panic: {}",
                    panic_message(panic_payload)
                ),
            })?
    })
    .join()
    .map_err(|panic_payload| crate::Error::GpuDeviceError {
        reason: format!(
            "GPU initialization thread panicked. Fix: inspect GPU initialization logs and driver state before retrying. Panic: {}",
            panic_message(panic_payload)
        ),
    })?
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}
