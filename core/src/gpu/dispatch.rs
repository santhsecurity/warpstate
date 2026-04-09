use wgpu::util::DeviceExt;

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::gpu::device::{self, GpuBufferPool};
use crate::gpu::readback;
use crate::pattern::PatternSet;
use crate::shader;
use crate::Match;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LiteralUniforms {
    pub input_len: u32,
    pub pattern_count: u32,
    pub max_matches: u32,
    pub hash_window_len: u32,
    pub reserved0: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RegexUniforms {
    pub input_len: u32,
    pub start_state: u32,
    pub max_matches: u32,
    pub class_count: u32,
    pub eoi_class: u32,
    pub _padding: [u32; 7],
    pub byte_classes: [[u32; 4]; 64],
}

#[derive(Debug)]
pub struct LiteralGpu {
    pub(crate) prefilter_pipeline: wgpu::ComputePipeline,
    pub(crate) verify_pipeline: wgpu::ComputePipeline,
    pub(crate) prefilter_layout: wgpu::BindGroupLayout,
    pub(crate) verify_layout: wgpu::BindGroupLayout,
    pub(crate) pattern_bytes_buf: wgpu::Buffer,
    pub(crate) pattern_offsets_buf: wgpu::Buffer,
    pub(crate) prefilter_prefix_meta_buf: wgpu::Buffer,
    pub(crate) prefilter_bucket_ranges_buf: wgpu::Buffer,
    pub(crate) prefilter_entries_buf: wgpu::Buffer,
    pub(crate) pattern_count: u32,
    pub(crate) pattern_ids: Vec<usize>,
    pub(crate) hash_window_len: u32,
}

#[derive(Debug)]
pub struct RegexGpu {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) transitions_buf: wgpu::Buffer,
    pub(crate) match_list_pointers_buf: wgpu::Buffer,
    pub(crate) match_lists_buf: wgpu::Buffer,
    pub(crate) pattern_lengths_buf: wgpu::Buffer,
    pub(crate) start_state: u32,
    pub(crate) class_count: u32,
    pub(crate) eoi_class: u32,
    pub(crate) byte_classes: [u32; 256],
    pub(crate) pattern_ids: Vec<usize>,
}

/// Specialized regex GPU with transitions embedded as shader constants.
/// Fewer bindings, better GPU compiler optimization, no buffer indirection.
#[derive(Debug)]
pub struct SpecializedRegexGpu {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) pattern_ids: Vec<usize>,
}

/// Uniforms for the specialized DFA shader (transitions are in constants).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SpecializedRegexUniforms {
    pub input_len: u32,
    pub max_matches: u32,
    pub _padding0: u32,
    pub _padding1: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn scan_literal_chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer_pool: &GpuBufferPool,
    pattern_set: &PatternSet,
    literal: &LiteralGpu,
    data: &[u8],
    base_offset: usize,
    max_matches: u32,
    max_input_size: usize,
) -> Result<Vec<Match>> {
    let input_len = super::to_u32_len(data.len(), max_input_size)?;
    let packed_input = device::pad_to_u32(data);
    let packed_bytes = device::packed_u32_as_bytes(&packed_input);
    let input_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let input_buf = buffer_pool.get_or_create(
        device,
        "warpstate literal input",
        packed_bytes.len() as u64,
        input_usage,
    );
    queue.write_buffer(&input_buf, 0, packed_bytes);

    // Candidate buffer: stores vec2<u32>(position, pattern_index) pairs.
    // Sized to max_matches entries (8 bytes each) for atomic append from prefilter.
    let candidate_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let candidate_buf = buffer_pool.get_or_create(
        device,
        "warpstate literal candidates",
        u64::from(max_matches) * 8, // vec2<u32> = 8 bytes
        candidate_usage,
    );
    // Atomic counter for candidate list append operations.
    // 8 bytes: [0..4] = candidate count, [4..8] = overflow flag.
    let candidate_count_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let candidate_count_buf = buffer_pool.get_or_create(
        device,
        "warpstate literal candidate count",
        8, // 2 x u32: count + overflow flag
        candidate_count_usage,
    );
    let match_buf_size = u64::from(max_matches) * 16;
    let match_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let match_buf = buffer_pool.get_or_create(
        device,
        "warpstate literal matches",
        match_buf_size,
        match_usage,
    );
    let count_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let count_buf = buffer_pool.get_or_create(device, "warpstate literal count", 8, count_usage);
    let prefilter_uniforms = LiteralUniforms {
        input_len,
        pattern_count: literal.pattern_count,
        max_matches,
        hash_window_len: literal.hash_window_len,
        reserved0: 0,
        _padding: [0; 3],
    };
    let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let uniform_size = std::mem::size_of::<LiteralUniforms>() as u64;
    let prefilter_uniform_buf = buffer_pool.get_or_create(
        device,
        "warpstate literal prefilter uniforms",
        uniform_size,
        uniform_usage,
    );
    queue.write_buffer(
        &prefilter_uniform_buf,
        0,
        bytemuck::bytes_of(&prefilter_uniforms),
    );
    let verify_uniform_buf = buffer_pool.get_or_create(
        device,
        "warpstate literal verify uniforms",
        uniform_size,
        uniform_usage,
    );
    queue.write_buffer(
        &verify_uniform_buf,
        0,
        bytemuck::bytes_of(&prefilter_uniforms),
    );

    let prefilter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warpstate literal prefilter bind group"),
        layout: &literal.prefilter_layout,
        entries: &[
            device::entry(0, &input_buf),
            device::entry(1, &literal.prefilter_prefix_meta_buf),
            device::entry(2, &literal.prefilter_bucket_ranges_buf),
            device::entry(3, &literal.prefilter_entries_buf),
            device::entry(4, &literal.pattern_offsets_buf),
            device::entry(5, &candidate_buf),
            device::entry(6, &candidate_count_buf),
            device::entry(7, &prefilter_uniform_buf),
        ],
    });
    let verify_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warpstate literal verify bind group"),
        layout: &literal.verify_layout,
        entries: &[
            device::entry(0, &input_buf),
            device::entry(1, &literal.pattern_bytes_buf),
            device::entry(2, &literal.pattern_offsets_buf),
            device::entry(3, &candidate_buf),
            device::entry(4, &candidate_count_buf),
            device::entry(5, &match_buf),
            device::entry(6, &count_buf),
            device::entry(7, &verify_uniform_buf),
        ],
    });

    let workgroups = compute_workgroups(device, input_len)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("warpstate literal encoder"),
    });
    encoder.clear_buffer(&candidate_buf, 0, None);
    encoder.clear_buffer(&candidate_count_buf, 0, None);
    encoder.clear_buffer(&count_buf, 0, None);

    // Staging buffer for reading back candidate count + overflow flag.
    let candidate_count_staging =
        device::readback_buffer(device, 8, "warpstate literal candidate count staging");
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warpstate literal prefilter pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&literal.prefilter_pipeline);
        pass.set_bind_group(0, &prefilter_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warpstate literal verify pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&literal.verify_pipeline);
        pass.set_bind_group(0, &verify_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let count_staging = device::readback_buffer(device, 8, "warpstate literal count staging");
    encoder.copy_buffer_to_buffer(&count_buf, 0, &count_staging, 0, 8);
    encoder.copy_buffer_to_buffer(&candidate_count_buf, 0, &candidate_count_staging, 0, 8);
    let match_staging =
        device::readback_buffer(device, match_buf_size, "warpstate literal match staging");
    encoder.copy_buffer_to_buffer(&match_buf, 0, &match_staging, 0, match_buf_size);
    queue.submit(Some(encoder.finish()));

    // Check for candidate buffer overflow from prefilter stage.
    // Overflow means some candidates were dropped — this would cause false negatives.
    readback::await_buffer_map(
        device,
        queue,
        &candidate_count_staging,
        "GPU candidate count buffer map",
    )
    .await?;

    let candidate_count_slice = candidate_count_staging.slice(..);
    let candidate_count_data = candidate_count_slice.get_mapped_range();
    let candidate_count_array: &[u32] = bytemuck::cast_slice(&candidate_count_data);
    let candidate_overflow = if candidate_count_array.len() >= 2 {
        candidate_count_array[1] != 0
    } else {
        false
    };
    drop(candidate_count_data);
    candidate_count_staging.unmap();

    let return_gpu_resources = || {
        buffer_pool.return_buffer(candidate_buf, candidate_usage);
        buffer_pool.return_buffer(candidate_count_buf, candidate_count_usage);
        buffer_pool.return_buffer(match_buf, match_usage);
        buffer_pool.return_buffer(count_buf, count_usage);
        buffer_pool.return_buffer(prefilter_uniform_buf, uniform_usage);
        buffer_pool.return_buffer(verify_uniform_buf, uniform_usage);
        buffer_pool.return_buffer(input_buf, input_usage);
    };

    if candidate_overflow {
        return_gpu_resources();

        let base_offset_u32 = u32::try_from(base_offset).map_err(|_| Error::InputTooLarge {
            bytes: usize::MAX,
            max_bytes: u32::MAX as usize,
        })?;

        let mut matches = pattern_set.scan(data)?;
        let literal_pattern_ids: HashSet<usize> = literal.pattern_ids.iter().copied().collect();
        for mat in &mut matches {
            let start_offset =
                base_offset_u32
                    .checked_add(mat.start)
                    .ok_or(Error::InputTooLarge {
                        bytes: usize::MAX,
                        max_bytes: u32::MAX as usize,
                    })?;
            let end_offset = base_offset_u32
                .checked_add(mat.end)
                .ok_or(Error::InputTooLarge {
                    bytes: usize::MAX,
                    max_bytes: u32::MAX as usize,
                })?;

            mat.start = start_offset;
            mat.end = end_offset;
        }
        matches.retain(|mat| literal_pattern_ids.contains(&(mat.pattern_id as usize)));

        return Ok(matches);
    }

    let result = readback::read_matches(
        device,
        queue,
        &count_staging,
        &match_staging,
        Some(&literal.pattern_ids),
        base_offset,
        max_matches,
        data.len(),
    )
    .await;

    return_gpu_resources();

    result
}

/// Compute the number of workgroups needed for an input, enforcing GPU limits.
pub(crate) fn compute_workgroups(device: &wgpu::Device, input_len: u32) -> Result<u32> {
    let raw = input_len.div_ceil(shader::WORKGROUP_SIZE);
    let max = device.limits().max_compute_workgroups_per_dimension;
    if raw > max {
        return Err(Error::GpuDeviceError {
            reason: format!(
                "input requires {raw} workgroups, exceeding device limit {max}. \
                 Fix: reduce chunk size or max input size."
            ),
        });
    }
    Ok(raw)
}

pub(crate) async fn scan_regex_chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer_pool: &GpuBufferPool,
    regex: &RegexGpu,
    data: &[u8],
    base_offset: usize,
    max_matches: u32,
    max_input_size: usize,
    max_regex_input_size: usize,
) -> Result<Vec<Match>> {
    if data.len() > max_regex_input_size {
        return Err(Error::InputTooLarge {
            bytes: data.len(),
            max_bytes: max_regex_input_size,
        });
    }
    let input_len = super::to_u32_len(data.len(), max_input_size)?;
    let packed_input = device::pad_to_u32(data);
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate regex input"),
        contents: device::packed_u32_as_bytes(&packed_input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let match_buf_size = u64::from(max_matches) * 16;
    let match_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let match_buf = buffer_pool.get_or_create(
        device,
        "warpstate regex matches",
        match_buf_size,
        match_usage,
    );
    let count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate regex count"),
        contents: bytemuck::cast_slice(&[0u32, 0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let mut byte_classes_chunks = [[0u32; 4]; 64];
    for i in 0..256usize {
        byte_classes_chunks[i / 4][i % 4] = regex.byte_classes[i];
    }
    let uniforms = RegexUniforms {
        input_len,
        start_state: regex.start_state,
        max_matches,
        class_count: regex.class_count,
        eoi_class: regex.eoi_class,
        _padding: [0; 7],
        byte_classes: byte_classes_chunks,
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate regex uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warpstate regex bind group"),
        layout: &regex.layout,
        entries: &[
            device::entry(0, &input_buf),
            device::entry(1, &regex.transitions_buf),
            device::entry(2, &regex.match_list_pointers_buf),
            device::entry(3, &regex.match_lists_buf),
            device::entry(4, &match_buf),
            device::entry(5, &count_buf),
            device::entry(6, &uniform_buf),
            device::entry(7, &regex.pattern_lengths_buf),
        ],
    });

    let workgroups = compute_workgroups(device, input_len)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("warpstate regex encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warpstate regex pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&regex.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let count_staging = device::readback_buffer(device, 8, "warpstate regex count staging");
    encoder.copy_buffer_to_buffer(&count_buf, 0, &count_staging, 0, 8);
    let match_staging =
        device::readback_buffer(device, match_buf_size, "warpstate regex match staging");
    encoder.copy_buffer_to_buffer(&match_buf, 0, &match_staging, 0, match_buf_size);
    queue.submit(Some(encoder.finish()));

    let result = readback::read_matches(
        device,
        queue,
        &count_staging,
        &match_staging,
        None,
        base_offset,
        max_matches,
        data.len(),
    )
    .await;

    buffer_pool.return_buffer(match_buf, match_usage);

    result
}

/// Scan using the specialized constant-embedded DFA shader.
/// This path has no transition buffer — all DFA data is baked into WGSL constants.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn scan_specialized_regex_chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer_pool: &GpuBufferPool,
    specialized: &SpecializedRegexGpu,
    data: &[u8],
    base_offset: usize,
    max_matches: u32,
    max_input_size: usize,
) -> Result<Vec<Match>> {
    let input_len = super::to_u32_len(data.len(), max_input_size)?;
    let packed_input = device::pad_to_u32(data);
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate specialized regex input"),
        contents: device::packed_u32_as_bytes(&packed_input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let match_buf_size = u64::from(max_matches) * 16;
    let match_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let match_buf = buffer_pool.get_or_create(
        device,
        "warpstate specialized regex matches",
        match_buf_size,
        match_usage,
    );
    let count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate specialized regex count"),
        contents: bytemuck::cast_slice(&[0u32, 0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let uniforms = SpecializedRegexUniforms {
        input_len,
        max_matches,
        _padding0: 0,
        _padding1: 0,
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate specialized regex uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warpstate specialized regex bind group"),
        layout: &specialized.layout,
        entries: &[
            device::entry(0, &input_buf),
            device::entry(1, &match_buf),
            device::entry(2, &count_buf),
            device::entry(3, &uniform_buf),
        ],
    });

    let workgroups = compute_workgroups(device, input_len)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("warpstate specialized regex encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warpstate specialized regex pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&specialized.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let count_staging =
        device::readback_buffer(device, 8, "warpstate specialized regex count staging");
    encoder.copy_buffer_to_buffer(&count_buf, 0, &count_staging, 0, 8);
    let match_staging = device::readback_buffer(
        device,
        match_buf_size,
        "warpstate specialized regex match staging",
    );
    encoder.copy_buffer_to_buffer(&match_buf, 0, &match_staging, 0, match_buf_size);
    queue.submit(Some(encoder.finish()));

    let result = readback::read_matches(
        device,
        queue,
        &count_staging,
        &match_staging,
        None,
        base_offset,
        max_matches,
        data.len(),
    )
    .await;

    buffer_pool.return_buffer(match_buf, match_usage);

    result
}
