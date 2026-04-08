use crate::gpu_smem::SmemDfaMatcher;
use crate::shader_smem;
use crate::GpuMatch;

/// Number of `u32` status slots in the GPU count buffer.
pub(super) const COUNT_BUFFER_U32S: u64 = 3;
/// Maximum number of matches to record.
pub(super) const MAX_MATCHES: u32 = 1_048_576;
/// Conservative shared-memory budget in bytes.
pub(super) const SMEM_BYTES: usize = 32 * 1024;

pub(super) struct PipelineState {
    pub(super) input_buf: wgpu::Buffer,
    pub(super) match_buf: wgpu::Buffer,
    pub(super) count_buf: wgpu::Buffer,
    pub(super) uniform_buf: wgpu::Buffer,
    pub(super) count_staging: wgpu::Buffer,
    pub(super) match_staging: wgpu::Buffer,
    pub(super) bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Uniforms {
    pub(super) input_len: u32,
    pub(super) start_state: u32,
    pub(super) max_matches: u32,
    pub(super) class_count: u32,
    pub(super) max_scan_depth: u32,
    pub(super) eoi_class: u32,
    pub(super) table_size: u32,
    pub(super) _padding0: u32,
    pub(super) byte_classes: [[u32; 4]; 64],
}

impl SmemDfaMatcher {
    /// Return the conservative maximum number of `u32` transition entries that fit in SMEM.
    #[must_use]
    pub const fn max_smem_entries() -> usize {
        shader_smem::MAX_SMEM_ENTRIES as usize
    }

    pub(super) fn allocate_buffers(
        device: &wgpu::Device,
        max_input_size: usize,
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
    ) {
        let padded_len = max_input_size.next_multiple_of(4);
        let match_buf_size = u64::from(MAX_MATCHES) * std::mem::size_of::<GpuMatch>() as u64;
        let count_buf_size = COUNT_BUFFER_U32S * std::mem::size_of::<u32>() as u64;

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smem input"),
            size: padded_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let match_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smem match buffer"),
            size: match_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smem count buffer"),
            size: count_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smem uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smem count staging"),
            size: count_buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smem match staging"),
            size: match_buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        (
            input_buf,
            match_buf,
            count_buf,
            uniform_buf,
            count_staging,
            match_staging,
        )
    }

    pub(super) fn assemble_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        transition_table: &wgpu::Buffer,
        match_list_pointers: &wgpu::Buffer,
        match_lists: &wgpu::Buffer,
        pattern_lengths: &wgpu::Buffer,
        state: &(
            wgpu::Buffer,
            wgpu::Buffer,
            wgpu::Buffer,
            wgpu::Buffer,
            wgpu::Buffer,
            wgpu::Buffer,
        ),
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smem dfa bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: transition_table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: match_list_pointers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: match_lists.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state.3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: pattern_lengths.as_entire_binding(),
                },
            ],
        })
    }
}

pub(super) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
