//! GPU backend — wgpu compute pipelines for literal and regex pattern matching.

/// GPU matcher construction helpers.
pub mod builder;
/// Shared-memory regex ensemble backend.
pub mod ensemble;
/// GPU device acquisition and shared buffer utilities.
pub mod device;
/// GPU scan dispatch paths for literal/regex kernels.
pub mod dispatch;
/// GPU buffer map/readback helpers.
pub mod readback;
#[cfg(test)]
#[cfg(not(miri))]
mod tests;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::config::AutoMatcherConfig;
use crate::error::{Error, Result};
use crate::pattern::PatternSet;
use crate::shader;
use crate::Match;

use self::builder::{build_literal_gpu, build_regex_gpu, build_specialized_regex_gpu};
use self::device::GpuBufferPool;
pub(crate) use self::device::{
    acquire_device, adapter_is_software, adapter_is_unsupported, SharedDeviceQueue,
};
use self::dispatch::{LiteralGpu, RegexGpu, SpecializedRegexGpu};

/// Returns `true` if the error may be resolved by recreating the GPU device.
fn is_recoverable_gpu_error(error: &Error) -> bool {
    match error {
        Error::GpuDeviceError { reason } => {
            reason.contains("sentinel")
                || reason.contains("device may be lost")
                || reason.contains("timed out")
                || reason.contains("Out of Memory")
                || reason.contains("out of memory")
                || reason.contains("OOM")
                || reason.contains("failed to create buffer")
                || reason.contains("allocation failed")
        }
        Error::BufferMapFailed => true,
        _ => false,
    }
}

/// Default maximum input size per GPU chunk in bytes (128 MB).
pub const DEFAULT_MAX_INPUT_SIZE: usize = 128 * 1024 * 1024;
/// Default chunk size for GPU scans.
pub const DEFAULT_CHUNK_SIZE: usize = DEFAULT_MAX_INPUT_SIZE;
/// Default overlap to preserve matches at chunk boundaries.
pub const DEFAULT_CHUNK_OVERLAP: usize = 4096;

/// Mutable GPU state that can be recreated after device loss.
///
/// The buffer pool is included here so that old buffers tied to a dead
/// device are dropped along with the old state, preventing contamination
/// of the new device's pool.
#[derive(Debug)]
struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    literal: Option<LiteralGpu>,
    regex: Option<RegexGpu>,
    specialized_regex: Option<SpecializedRegexGpu>,
    buffer_pool: GpuBufferPool,
}

/// GPU-accelerated pattern matcher using wgpu compute shaders.
#[derive(Debug)]
pub struct GpuMatcher {
    state: ArcSwap<GpuState>,
    device_needs_recreation: AtomicBool,
    patterns: PatternSet,
    max_input_size: usize,
    max_regex_input_size: usize,
    chunk_size: usize,
    chunk_overlap: usize,
    max_matches: u32,
    hard_input_limit: bool,
    config: AutoMatcherConfig,
}

impl GpuMatcher {
    /// Create a new GPU matcher using default configuration.
    pub async fn new(patterns: &PatternSet) -> Result<Self> {
        Self::with_config(patterns, AutoMatcherConfig::default()).await
    }

    /// Legacy constructor with a strict input size cap.
    pub async fn with_options(patterns: &PatternSet, max_input_size: usize) -> Result<Self> {
        let mut matcher = Self::with_config(
            patterns,
            AutoMatcherConfig::new()
                .gpu_max_input_size(max_input_size)
                .chunk_size(max_input_size),
        )
        .await?;
        matcher.hard_input_limit = true;
        Ok(matcher)
    }

    /// Create a GPU matcher with explicit configuration.
    pub async fn with_config(patterns: &PatternSet, config: AutoMatcherConfig) -> Result<Self> {
        let device_queue = acquire_device().await?;
        Self::from_device(&device_queue, patterns, config)
    }

    /// Create a GPU matcher from an existing single-GPU device/queue pair.
    pub fn from_device(
        device_queue: &SharedDeviceQueue,
        patterns: &PatternSet,
        config: AutoMatcherConfig,
    ) -> Result<Self> {
        Self::build_from_device(
            device_queue.0.clone(),
            device_queue.1.clone(),
            patterns,
            config,
        )
    }

    pub(crate) async fn from_adapter(
        patterns: &PatternSet,
        adapter: &wgpu::Adapter,
        config: AutoMatcherConfig,
    ) -> Result<Option<Self>> {
        if adapter_is_unsupported(&adapter.get_info()) {
            return Ok(None);
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| Error::GpuDeviceError {
                reason: e.to_string(),
            })?;

        Ok(Some(Self::build_from_device(
            device, queue, patterns, config,
        )?))
    }

    fn build_from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        patterns: &PatternSet,
        config: AutoMatcherConfig,
    ) -> Result<Self> {
        let limits = device.limits();
        let max_storage = limits.max_storage_buffer_binding_size as usize;
        let max_buffer = limits.max_buffer_size as usize;
        let max_workgroups = limits.max_compute_workgroups_per_dimension as usize;
        let max_input_for_workgroups =
            max_workgroups.saturating_mul(shader::WORKGROUP_SIZE as usize);
        let hard_u32_limit = u32::MAX as usize;
        let effective_max_input = config
            .configured_gpu_max_input_size()
            .min(max_storage)
            .min(max_buffer)
            .min(max_input_for_workgroups)
            .min(hard_u32_limit);

        let chunk_overlap = config.configured_chunk_overlap().max(
            patterns
                .ir()
                .offsets
                .iter()
                .map(|&(_, len)| len as usize)
                .max()
                .unwrap_or(0),
        );
        // Ensure a chunk plus its overlap never exceeds the device limit,
        // preventing wgpu buffer allocation panics at chunk boundaries.
        let effective_chunk = config
            .configured_chunk_size()
            .max(1)
            .min(effective_max_input)
            .min(effective_max_input.saturating_sub(chunk_overlap).max(1));

        let literal = build_literal_gpu(&device, patterns)?;
        // Try specialized shader first (DFA constants in WGSL), fall back to buffer-based
        let specialized_regex = build_specialized_regex_gpu(&device, patterns)?;
        let regex = if specialized_regex.is_some() {
            None // Don't build buffer-based if specialized succeeds
        } else {
            build_regex_gpu(&device, &queue, patterns)?
        };

        let state = GpuState {
            device,
            queue,
            literal,
            regex,
            specialized_regex,
            buffer_pool: GpuBufferPool::default(),
        };

        // The regex DFA shader hardcodes a 16MB scan-depth cap. Allowing larger
        // inputs silently produces false negatives. Clamp the API limit to match
        // the shader guarantee.
        let max_regex_input_size = config
            .configured_gpu_max_regex_input_size()
            .min(crate::config::DEFAULT_MAX_REGEX_INPUT_SIZE);

        Ok(Self {
            state: ArcSwap::from(Arc::new(state)),
            device_needs_recreation: AtomicBool::new(false),
            patterns: patterns.clone(),
            max_input_size: effective_max_input,
            max_regex_input_size,
            chunk_size: effective_chunk,
            chunk_overlap,
            max_matches: config.configured_max_matches(),
            hard_input_limit: false,
            config,
        })
    }

    /// Scan input data for pattern matches on the GPU.
    pub async fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        if data.is_empty() {
            // Regex patterns may match empty input (e.g., `.*`). The GPU shaders
            // dispatch zero workgroups on empty input, so fall back to CPU for
            // correct empty-input semantics.
            return self.fallback_cpu_scan(data);
        }
        if self.has_regex_pipeline() && data.len() > self.max_regex_input_size {
            return Err(Error::InputTooLarge {
                bytes: data.len(),
                max_bytes: self.max_regex_input_size,
            });
        }
        if self.hard_input_limit && data.len() > self.max_input_size {
            return Err(Error::InputTooLarge {
                bytes: data.len(),
                max_bytes: self.max_input_size,
            });
        }

        let result = self.scan_once(data).await;

        match result {
            Ok(matches) => Ok(matches),
            Err(e) if is_recoverable_gpu_error(&e) => {
                self.device_needs_recreation.store(true, Ordering::SeqCst);
                // Only the thread that successfully swaps the flag recreates.
                if self.device_needs_recreation.swap(false, Ordering::SeqCst) {
                    if self.recreate_device().await.is_err() {
                        self.device_needs_recreation.store(true, Ordering::SeqCst);
                        return self.fallback_cpu_scan(data);
                    }
                }
                self.scan_once(data).await
            }
            Err(e) => Err(e),
        }
    }

    async fn scan_once(&self, data: &[u8]) -> Result<Vec<Match>> {
        // If another thread detected device loss, try to recreate before scanning.
        // swap(false) ensures only one thread actually performs recreation.
        if self.device_needs_recreation.swap(false, Ordering::SeqCst) {
            if self.recreate_device().await.is_err() {
                self.device_needs_recreation.store(true, Ordering::SeqCst);
                return Err(Error::GpuDeviceError {
                    reason: "GPU device recreation failed".to_string(),
                });
            }
        }

        if data.len() <= self.chunk_size && data.len() <= self.max_input_size {
            self.scan_chunk(data, 0).await
        } else {
            let mut all_matches = Vec::new();
            let mut start = 0usize;
            loop {
                let nominal_end = start.saturating_add(self.chunk_size).min(data.len());
                let chunk_end = nominal_end
                    .saturating_add(self.chunk_overlap)
                    .min(data.len());
                match self.scan_chunk(&data[start..chunk_end], start).await {
                    Ok(mut chunk_matches) => {
                        chunk_matches.retain(|m| (m.start as usize) < nominal_end);
                        all_matches.extend(chunk_matches);
                    }
                    Err(e) => break Err(e),
                }
                if nominal_end == data.len() {
                    break Ok(all_matches);
                }
                start = nominal_end;
            }
        }
    }

    /// Synchronous scan for callers without an async runtime.
    ///
    /// Uses `pollster::block_on` — zero-allocation, no thread pool creation.
    /// At scale (40K files/hr), creating a tokio runtime per file would exhaust
    /// the blocking thread pool. pollster avoids this entirely.
    pub fn scan_blocking(&self, input: &[u8]) -> Result<Vec<Match>> {
        pollster::block_on(self.scan(input))
    }

    /// Returns a clone of the underlying wgpu device and queue.
    ///
    /// This enables other GPU-accelerated crates (e.g., `gputokenize`, `rulefire`)
    /// to share the same GPU context without creating a second device.
    #[must_use]
    pub fn gpu_device_queue(&self) -> (wgpu::Device, wgpu::Queue) {
        let state = self.state.load_full();
        (state.device.clone(), state.queue.clone())
    }

    pub(crate) fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    /// Recreate the wgpu device, queue, and all pipelines.
    ///
    /// Called automatically when a scan detects device loss. Thread-safe:
    /// only the thread that successfully swaps the flag from true→false
    /// performs the recreation; others observe the new state on retry.
    async fn recreate_device(&self) -> Result<()> {
        let device_queue = acquire_device().await?;
        let (device, queue) = (device_queue.0.clone(), device_queue.1.clone());

        let literal = build_literal_gpu(&device, &self.patterns)?;
        let specialized_regex = build_specialized_regex_gpu(&device, &self.patterns)?;
        let regex = if specialized_regex.is_some() {
            None
        } else {
            build_regex_gpu(&device, &queue, &self.patterns)?
        };

        let state = GpuState {
            device,
            queue,
            literal,
            regex,
            specialized_regex,
            buffer_pool: GpuBufferPool::default(),
        };

        self.state.store(Arc::new(state));
        self.device_needs_recreation.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Fall back to CPU scanning when the GPU is permanently unavailable.
    fn fallback_cpu_scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        self.patterns.scan(data)
    }

    fn has_regex_pipeline(&self) -> bool {
        let state = self.state.load_full();
        state.regex.is_some() || state.specialized_regex.is_some()
    }

    async fn scan_chunk(&self, data: &[u8], base_offset: usize) -> Result<Vec<Match>> {
        let state = self.state.load_full();
        let mut matches = Vec::new();
        if let Some(literal) = &state.literal {
            matches.extend(
                self.scan_literal_chunk(
                    &state.device,
                    &state.queue,
                    &state.buffer_pool,
                    literal,
                    data,
                    base_offset,
                )
                .await?,
            );
        }
        // Prefer specialized regex (constants in shader) over buffer-based
        if let Some(specialized) = &state.specialized_regex {
            matches.extend(
                dispatch::scan_specialized_regex_chunk(
                    &state.device,
                    &state.queue,
                    &state.buffer_pool,
                    specialized,
                    data,
                    base_offset,
                    self.max_matches,
                    self.max_input_size,
                )
                .await?,
            );
        } else if let Some(regex) = &state.regex {
            matches.extend(
                self.scan_regex_chunk(
                    &state.device,
                    &state.queue,
                    &state.buffer_pool,
                    regex,
                    data,
                    base_offset,
                )
                .await?,
            );
        }
        matches.sort_unstable();
        Ok(matches)
    }

    async fn scan_literal_chunk(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_pool: &GpuBufferPool,
        literal: &LiteralGpu,
        data: &[u8],
        base_offset: usize,
    ) -> Result<Vec<Match>> {
        dispatch::scan_literal_chunk(
            device,
            queue,
            buffer_pool,
            &self.patterns,
            literal,
            data,
            base_offset,
            self.max_matches,
            self.max_input_size,
        )
        .await
    }

    async fn scan_regex_chunk(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_pool: &GpuBufferPool,
        regex: &RegexGpu,
        data: &[u8],
        base_offset: usize,
    ) -> Result<Vec<Match>> {
        dispatch::scan_regex_chunk(
            device,
            queue,
            buffer_pool,
            &self.patterns,
            regex,
            data,
            base_offset,
            self.max_regex_input_size,
        )
        .await
    }
}

pub(crate) fn to_u32_len(len: usize, max_bytes: usize) -> Result<u32> {
    let hard_limit = u32::MAX as usize;
    len.try_into().map_err(|_| Error::InputTooLarge {
        bytes: len,
        max_bytes: max_bytes.min(hard_limit),
    })
}

/// Check if the system GPU is a software renderer (llvmpipe, lavapipe, etc.)
/// that has known issues with regex DFA shaders.
///
/// This is a public API for tests to determine if they should skip GPU regex tests.
/// Returns `true` if no GPU adapter is available or if the adapter is a known
/// software renderer.
///
/// # Example
/// ```
/// use warpstate::gpu::is_software_gpu;
///
/// if is_software_gpu() {
///     // Skip test or use CPU fallback
///     return;
/// }
/// ```
pub fn is_software_gpu() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));

    match adapter {
        Some(a) => adapter_is_software(&a.get_info()),
        None => true, // No GPU available, treat as "software" (skip tests)
    }
}
