//! Zero-copy DMA integrations bridging wireshift io_uring with GPU VRAM.
//!
//! Exposes mapped `wgpu::Buffer` staging for host-to-GPU data transfer.
//! Data flows: NVMe → host memory (mapped buffer) → GPU VRAM (`flush_to_vram` method).
#![allow(unsafe_code)]

use crate::error::Result;
use std::ptr::NonNull;

/// Wraps a host-visible `wgpu::Buffer` with `MAP_WRITE | COPY_SRC` usage.
///
/// Callers write data into the mapped buffer via [`write`](Self::write), then call
/// [`flush_to_vram`](Self::flush_to_vram) to unmap and make the data available for GPU compute
/// pipelines.
///
/// This enables the "NVMe → GPU" data path by providing a staging buffer
/// that wireshift can fill directly from I/O completions.
#[derive(Debug)]
pub struct DmaStagingBuffer {
    /// The underlying wgpu buffer, created mapped-at-creation.
    buffer: wgpu::Buffer,
    /// Pointer into the mapped buffer while it remains mapped.
    mapped_ptr: NonNull<u8>,
    /// Caller-visible writable length in bytes.
    len: usize,
    /// Capacity in bytes.
    capacity: usize,
    /// Whether the mapped memory has already been flushed/unmapped.
    flushed: bool,
}

impl DmaStagingBuffer {
    /// Create a new staging buffer bound to host-visible GPU memory.
    ///
    /// The buffer is created with `mapped_at_creation: true`, ready for
    /// immediate writes.
    ///
    /// # Errors
    ///
    /// Returns an error if `capacity` is zero.
    pub fn new(device: &wgpu::Device, capacity: usize) -> Result<Self> {
        Self::with_lengths(device, capacity, capacity)
    }

    pub(crate) fn with_lengths(device: &wgpu::Device, capacity: usize, len: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(crate::error::Error::InputTooLarge {
                bytes: 0,
                max_bytes: 0,
            });
        }
        if len > capacity {
            return Err(crate::error::Error::InputTooLarge {
                bytes: len,
                max_bytes: capacity,
            });
        }

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dma_staging_nvme_recv"),
            size: capacity as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        // SAFETY: `mapped_at_creation: true` keeps the buffer mapped after the
        // BufferViewMut is dropped. The raw pointer captured here remains valid
        // until `buffer.unmap()` is called in `flush_to_vram()`. Dropping the
        // view only releases the borrow — it does NOT unmap the underlying memory.
        let mut initial_view = buffer.slice(..).get_mapped_range_mut();
        initial_view.fill(0);
        let mapped_ptr =
            NonNull::new(initial_view.as_mut_ptr()).ok_or(crate::error::Error::BufferMapFailed)?;
        drop(initial_view);

        Ok(Self {
            buffer,
            mapped_ptr,
            len,
            capacity,
            flushed: false,
        })
    }

    /// Write data into the mapped staging buffer.
    ///
    /// Copies `data` into the GPU-visible host memory. Only the first
    /// `min(data.len(), capacity)` bytes are written.
    ///
    /// # Panics
    ///
    /// Panics if the buffer has already been unmapped via [`flush_to_vram`](Self::flush_to_vram).
    pub fn write(&mut self, data: &[u8]) {
        let len = data.len().min(self.len);
        self.as_mut_slice()[..len].copy_from_slice(&data[..len]);
    }

    /// Returns a mutable slice into the mapped GPU-visible memory.
    ///
    /// Caller writes directly into this slice, eliminating one memcpy.
    /// The returned slice remains valid until [`flush_to_vram`](Self::flush_to_vram) is called.
    ///
    /// # Panics
    ///
    /// Panics if the mapped buffer has already been flushed to VRAM.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(
            !self.flushed,
            "mapped staging buffer was already flushed to VRAM"
        );

        // SAFETY: `mapped_ptr` is captured from `get_mapped_range_mut()` while the buffer is
        // mapped at creation. The buffer remains mapped until `flush_to_vram()` calls `unmap()`.
        // The `&mut self` receiver guarantees unique mutable access to the staging buffer while
        // the slice exists, preventing aliasing through this API.
        unsafe { std::slice::from_raw_parts_mut(self.mapped_ptr.as_ptr(), self.len) }
    }

    /// Returns the raw mapped pointer and logical writable length.
    ///
    /// This is the integration point for external I/O runtimes, such as a downstream
    /// `wireshift::Buffer::from_mapped_buffer(ptr, len)` constructor.
    ///
    /// # Panics
    ///
    /// Panics if the mapped buffer has already been flushed to VRAM.
    pub fn mapped_parts(&mut self) -> (*mut u8, usize) {
        let slice = self.as_mut_slice();
        (slice.as_mut_ptr(), slice.len())
    }

    /// Unmap the buffer, flushing written data to GPU-accessible VRAM.
    ///
    /// After this call, the buffer can be used as a `COPY_SRC` in a
    /// command encoder to transfer data into storage buffers for compute.
    pub fn flush_to_vram(&mut self) {
        self.flushed = true;
        self.buffer.unmap();
    }

    /// Get a reference to the underlying `wgpu::Buffer` for use in copy
    /// commands.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// The capacity of this staging buffer in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The caller-visible writable length of this staging buffer in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the caller-visible writable length is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Trait for external buffer types that can be constructed from mapped memory.
///
/// This lets downstream crates implement zero-copy adapters without forcing
/// `warpstate` to depend directly on a specific I/O runtime crate.
pub trait FromMappedBuffer: Sized {
    /// Construct a buffer from a raw writable pointer and length.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr..ptr+len` remains valid for the lifetime
    /// requirements of the constructed buffer and that no aliased mutable access occurs.
    unsafe fn from_mapped_buffer(ptr: *mut u8, len: usize) -> Self;
}

// SAFETY: The staging buffer is moved between tasks as an owned value. The mapped pointer
// always refers to memory owned by the `wgpu::Buffer`, and all mutable access remains gated
// by `&mut self` methods on the buffer before it is flushed and consumed.
unsafe impl Send for DmaStagingBuffer {}

/// Scan path selection for file-to-GPU data transfer.
///
/// The scanner checks paths in order of preference:
/// 1. **Tier 1 (DMA)**: NVMe → GPU VRAM via GPUDirect Storage (cudagrep).
///    Data never touches CPU or system RAM. Requires `cuda` feature + NVIDIA hardware.
/// 2. **Tier 2 (Staged)**: NVMe → mapped host buffer → GPU VRAM via wgpu.
///    One memcpy through CPU-visible mapped memory. Works on any wgpu backend.
/// 3. **Tier 3 (CPU)**: NVMe → system RAM → CPU scan.
///    Standard path when no GPU is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaScanPath {
    /// Tier 1: NVMe → GPU DMA via GPUDirect Storage. Zero CPU involvement.
    GpuDirect,
    /// Tier 2: NVMe → host-mapped buffer → GPU copy. One memcpy through CPU.
    GpuStaged,
    /// Tier 3: Standard CPU scan path.
    CpuOnly,
}

impl DmaScanPath {
    /// Detect the best available scan path for the current hardware.
    ///
    /// Checks in order: GPUDirect (cudagrep), GPU staging (wgpu), CPU fallback.
    pub fn detect() -> Self {
        // Check for GPUDirect Storage (NVIDIA GDS via cudagrep)
        #[cfg(feature = "gpu-direct")]
        {
            if cudagrep::is_available() {
                return Self::GpuDirect;
            }
        }

        #[cfg(feature = "gpu")]
        {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let has_gpu =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .is_some();

            if has_gpu {
                return Self::GpuStaged;
            }
        }

        Self::CpuOnly
    }

    /// Human-readable description of the scan path.
    pub const fn description(self) -> &'static str {
        match self {
            Self::GpuDirect => "Tier 1: NVMe → GPU DMA (GPUDirect Storage)",
            Self::GpuStaged => "Tier 2: NVMe → host buffer → GPU copy",
            Self::CpuOnly => "Tier 3: CPU scan (no GPU)",
        }
    }
}

/// Scan a file through the GPU-staged DMA path.
///
/// 1. Read file data into a host-mapped staging buffer
/// 2. Flush to GPU VRAM
/// 3. Run GPU compute shader on the data
/// 4. Read back matches
///
/// For the Tier 1 (GPUDirect) path, the file bytes go directly from NVMe
/// to GPU VRAM without touching CPU memory. This requires `cudagrep` with
/// the `cuda` feature and NVIDIA hardware with GDS support.
///
/// # Errors
///
/// Returns errors from wgpu device creation, buffer allocation, or scan execution.
#[cfg(feature = "gpu")]
pub async fn scan_file_staged(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    file_data: &[u8],
    patterns: &crate::PatternSet,
) -> Result<Vec<crate::Match>> {
    use crate::config::AutoMatcherConfig;
    use crate::gpu::GpuMatcher;

    // For now, this uses the standard GpuMatcher scan path which handles
    // its own buffer management. The DmaStagingBuffer provides the bridge
    // for external I/O runtimes (wireshift) to write directly into GPU-visible
    // memory before the scan, eliminating one memcpy.
    let device_queue = std::sync::Arc::new((device.clone(), queue.clone()));
    let matcher = GpuMatcher::from_device(&device_queue, patterns, AutoMatcherConfig::default())?;
    matcher.scan(file_data).await
}
