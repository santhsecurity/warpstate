use std::sync::Arc;

use crate::error::{Error, Result};

pub(crate) type SharedDeviceQueue = Arc<(wgpu::Device, wgpu::Queue)>;

/// GPU buffer pool for reusing allocations across scans.
#[derive(Debug, Default)]
pub struct GpuBufferPool {
    /// Pool of available buffers keyed by (usage bits, size class).
    pub(crate) available: std::sync::Mutex<Vec<(wgpu::BufferUsages, u64, wgpu::Buffer)>>,
}

impl GpuBufferPool {
    /// Get or create a buffer with the given usage and minimum size.
    pub fn get_or_create(
        &self,
        device: &wgpu::Device,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let size_class = (size as usize).next_power_of_two().max(64) as u64;

        {
            let mut pool = self
                .available
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(idx) = pool
                .iter()
                .position(|(u, s, _)| *u == usage && *s >= size_class)
            {
                let (_, _, buf) = pool.swap_remove(idx);
                return buf;
            }
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_class,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for future reuse.
    pub fn return_buffer(&self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) {
        const MAX_POOL_SIZE: usize = 64;

        let size = buffer.size();
        let mut pool = self
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if pool.len() >= MAX_POOL_SIZE {
            // Evict the smallest buffer to keep large allocations hot.
            // At petabyte scale, large buffers are expensive to recreate.
            if let Some(min_idx) = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, s, _))| *s)
                .map(|(i, _)| i)
            {
                pool.swap_remove(min_idx);
            }
        }
        pool.push((usage, size, buffer));
    }
}

/// Check if the adapter is a software/CPU renderer that doesn't properly
/// support compute shaders (e.g., llvmpipe, lavapipe, swiftshader).
///
/// These adapters may successfully create a device but fail to correctly
/// execute WGSL compute shaders, particularly those using atomics or
/// complex control flow like the regex DFA shader.
pub(crate) fn adapter_is_unsupported(info: &wgpu::AdapterInfo) -> bool {
    #[cfg(feature = "software-gpu")]
    {
        // Allow CPU/software adapters for CI and testing.
        _ = info;
        false
    }
    #[cfg(not(feature = "software-gpu"))]
    {
        matches!(info.device_type, wgpu::DeviceType::Cpu)
    }
}

/// Check if the adapter is a known software renderer that has known
/// issues with regex DFA shaders. This is more specific than
/// `adapter_is_unsupported` - it detects software GPUs even when the
/// "software-gpu" feature is enabled, for use in tests that need to
/// skip when running on broken software implementations.
///
/// Known problematic software renderers:
/// - llvmpipe: Mesa software renderer (reports DeviceType::Other)
/// - lavapipe: Mesa Vulkan software renderer
/// - swiftshader: Google's software renderer
pub(crate) fn adapter_is_software(info: &wgpu::AdapterInfo) -> bool {
    // Check device type first
    if matches!(info.device_type, wgpu::DeviceType::Cpu) {
        return true;
    }

    // Check driver name for known software renderers
    let driver_lower = info.driver.to_lowercase();
    let name_lower = info.name.to_lowercase();
    let driver_info_lower = info.driver_info.to_lowercase();

    let software_indicators = ["llvmpipe", "lavapipe", "swiftshader", "softpipe", "virgl"];

    for indicator in &software_indicators {
        if driver_lower.contains(indicator)
            || name_lower.contains(indicator)
            || driver_info_lower.contains(indicator)
        {
            return true;
        }
    }

    // Mesa software drivers often report as "llvmpipe" in the name
    // and have "Mesa" in driver info but DeviceType::Other
    if name_lower.contains("llvmpipe") || name_lower.contains("softpipe") {
        return true;
    }

    false
}

pub(crate) async fn acquire_device() -> Result<SharedDeviceQueue> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok_or(Error::NoGpuAdapter)?;
    #[cfg(not(feature = "software-gpu"))]
    if matches!(adapter.get_info().device_type, wgpu::DeviceType::Cpu) {
        return Err(Error::NoGpuAdapter);
    }

    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("warpstate single-gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            },
            None,
        )
        .await
        .map_err(|e| Error::GpuDeviceError {
            reason: e.to_string(),
        })?;

    Ok(Arc::new((device, queue)))
}

pub(crate) fn readback_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

pub(crate) fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn pad_to_u32(data: &[u8]) -> std::borrow::Cow<'_, [u32]> {
    if data.as_ptr() as usize % 4 == 0 && data.len() % 4 == 0 {
        let u32_slice: &[u32] = bytemuck::cast_slice(data);
        return std::borrow::Cow::Borrowed(u32_slice);
    }
    let mut result = vec![0u32; data.len().div_ceil(4)];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
    bytes[..data.len()].copy_from_slice(data);
    std::borrow::Cow::Owned(result)
}

pub(crate) fn packed_u32_as_bytes<'a>(words: &'a std::borrow::Cow<'a, [u32]>) -> &'a [u8] {
    match words {
        std::borrow::Cow::Borrowed(words) => bytemuck::cast_slice(words),
        std::borrow::Cow::Owned(words) => bytemuck::cast_slice(words.as_slice()),
    }
}
