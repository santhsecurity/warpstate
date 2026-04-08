//! GPU backend — wgpu compute pipelines for literal and regex pattern matching.

pub mod builder;
pub mod device;
pub mod dispatch;
pub mod readback;
#[cfg(test)]
#[cfg(not(miri))]
mod tests;

use crate::config::AutoMatcherConfig;
use crate::error::{Error, Result};
use crate::pattern::PatternSet;
use crate::Match;

use self::builder::{build_literal_gpu, build_regex_gpu, build_specialized_regex_gpu};
use self::device::GpuBufferPool;
pub(crate) use self::device::{
    acquire_device, adapter_is_software, adapter_is_unsupported, SharedDeviceQueue,
};
use self::dispatch::{LiteralGpu, RegexGpu, SpecializedRegexGpu};

/// Default maximum input size per GPU chunk in bytes (128 MB).
pub const DEFAULT_MAX_INPUT_SIZE: usize = 128 * 1024 * 1024;
/// Default chunk size for GPU scans.
pub const DEFAULT_CHUNK_SIZE: usize = DEFAULT_MAX_INPUT_SIZE;
/// Default overlap to preserve matches at chunk boundaries.
pub const DEFAULT_CHUNK_OVERLAP: usize = 4096;

/// GPU-accelerated pattern matcher using wgpu compute shaders.
#[derive(Debug)]
pub struct GpuMatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    patterns: PatternSet,
    literal: Option<LiteralGpu>,
    regex: Option<RegexGpu>,
    /// Specialized regex with DFA constants baked into WGSL. Preferred over `regex`
    /// when the DFA is small enough (< 16384 transitions).
    specialized_regex: Option<SpecializedRegexGpu>,
    max_input_size: usize,
    max_regex_input_size: usize,
    chunk_size: usize,
    chunk_overlap: usize,
    max_matches: u32,
    hard_input_limit: bool,
    /// Buffer pool for reusing GPU allocations across scans.
    buffer_pool: GpuBufferPool,
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
        let effective_max_input = config.configured_gpu_max_input_size().min(max_storage);
        let effective_chunk = config
            .configured_chunk_size()
            .max(1)
            .min(effective_max_input);

        let literal = build_literal_gpu(&device, patterns)?;
        // Try specialized shader first (DFA constants in WGSL), fall back to buffer-based
        let specialized_regex = build_specialized_regex_gpu(&device, patterns)?;
        let regex = if specialized_regex.is_some() {
            None // Don't build buffer-based if specialized succeeds
        } else {
            build_regex_gpu(&device, patterns)?
        };
        let chunk_overlap = config.configured_chunk_overlap().max(
            patterns
                .ir()
                .offsets
                .iter()
                .map(|&(_, len)| len as usize)
                .max()
                .unwrap_or(0),
        );

        Ok(Self {
            device,
            queue,
            patterns: patterns.clone(),
            literal,
            regex,
            specialized_regex,
            max_input_size: effective_max_input,
            max_regex_input_size: config.configured_gpu_max_regex_input_size(),
            chunk_size: effective_chunk,
            chunk_overlap,
            max_matches: config.configured_max_matches(),
            hard_input_limit: false,
            buffer_pool: GpuBufferPool::default(),
        })
    }

    /// Scan input data for pattern matches on the GPU.
    pub async fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if (self.regex.is_some() || self.specialized_regex.is_some())
            && data.len() > self.max_regex_input_size
        {
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

        if data.len() <= self.chunk_size && data.len() <= self.max_input_size {
            return self.scan_chunk(data, 0).await;
        }

        let mut all_matches = Vec::new();
        let mut start = 0usize;
        while start < data.len() {
            let nominal_end = start.saturating_add(self.chunk_size).min(data.len());
            let chunk_end = nominal_end
                .saturating_add(self.chunk_overlap)
                .min(data.len());
            let mut chunk_matches = self.scan_chunk(&data[start..chunk_end], start).await?;
            chunk_matches.retain(|m| (m.start as usize) < nominal_end);
            all_matches.extend(chunk_matches);
            if nominal_end == data.len() {
                break;
            }
            start = nominal_end;
        }

        all_matches.sort_unstable();
        all_matches.dedup_by(|left, right| {
            left.pattern_id == right.pattern_id
                && left.start == right.start
                && left.end == right.end
        });
        Ok(all_matches)
    }

    /// Synchronous scan for callers without an async runtime.
    pub fn scan_blocking(&self, input: &[u8]) -> Result<Vec<Match>> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::GpuDeviceError {
                reason: format!("tokio runtime creation failed: {e}"),
            })?
            .block_on(self.scan(input))
    }

    /// Returns a reference to the underlying wgpu device and queue.
    ///
    /// This enables other GPU-accelerated crates (e.g., `gputokenize`, `rulefire`)
    /// to share the same GPU context without creating a second device.
    #[must_use]
    pub fn gpu_device_queue(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.device, &self.queue)
    }

    pub(crate) fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    async fn scan_chunk(&self, data: &[u8], base_offset: usize) -> Result<Vec<Match>> {
        let mut matches = Vec::new();
        if let Some(literal) = &self.literal {
            matches.extend(self.scan_literal_chunk(literal, data, base_offset).await?);
        }
        // Prefer specialized regex (constants in shader) over buffer-based
        if let Some(specialized) = &self.specialized_regex {
            matches.extend(
                dispatch::scan_specialized_regex_chunk(
                    &self.device,
                    &self.queue,
                    &self.buffer_pool,
                    specialized,
                    data,
                    base_offset,
                    self.max_matches,
                    self.max_input_size,
                )
                .await?,
            );
        } else if let Some(regex) = &self.regex {
            matches.extend(self.scan_regex_chunk(regex, data, base_offset).await?);
        }
        matches.sort_unstable();
        Ok(matches)
    }

    async fn scan_literal_chunk(
        &self,
        literal: &LiteralGpu,
        data: &[u8],
        base_offset: usize,
    ) -> Result<Vec<Match>> {
        dispatch::scan_literal_chunk(
            &self.device,
            &self.queue,
            &self.buffer_pool,
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
        regex: &RegexGpu,
        data: &[u8],
        base_offset: usize,
    ) -> Result<Vec<Match>> {
        dispatch::scan_regex_chunk(
            &self.device,
            &self.queue,
            &self.buffer_pool,
            regex,
            data,
            base_offset,
            self.max_matches,
            self.max_input_size,
            self.max_regex_input_size,
        )
        .await
    }
}

pub(crate) fn to_u32_len(len: usize, max_bytes: usize) -> Result<u32> {
    len.try_into().map_err(|_| Error::InputTooLarge {
        bytes: len,
        max_bytes,
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
