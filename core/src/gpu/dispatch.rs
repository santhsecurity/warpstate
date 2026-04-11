use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::gpu::device::{self, GpuBufferPool};
use crate::gpu::ensemble::{EnsembleRegexMatcher, SmallDfa};
use crate::gpu::readback;
use crate::gpu::shader;
use crate::pattern::PatternSet;
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

/// GPU resources for literal prefilter+verify matching.
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

/// GPU resources for regex DFA matching.
#[derive(Debug)]
pub struct RegexGpu {
    pub(crate) matcher: EnsembleRegexMatcher,
    pub(crate) small_dfas: Vec<SmallDfa>,
}

impl Drop for LiteralGpu {
    fn drop(&mut self) {
        self.pattern_bytes_buf.destroy();
        self.pattern_offsets_buf.destroy();
        self.prefilter_prefix_meta_buf.destroy();
        self.prefilter_bucket_ranges_buf.destroy();
        self.prefilter_entries_buf.destroy();
    }
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
    let workgroups = compute_workgroups(device, input_len)?;
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
        u64::from(max_matches)
            .checked_mul(8)
            .ok_or_else(|| Error::InputTooLarge {
                bytes: usize::MAX,
                max_bytes: max_input_size,
            })?, // vec2<u32> = 8 bytes
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
    let match_buf_size =
        u64::from(max_matches)
            .checked_mul(16)
            .ok_or_else(|| Error::InputTooLarge {
                bytes: usize::MAX,
                max_bytes: max_input_size,
            })?;
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
        pass.dispatch_workgroups(workgroups.x, workgroups.y, workgroups.z);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warpstate literal verify pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&literal.verify_pipeline);
        pass.set_bind_group(0, &verify_bind_group, &[]);
        pass.dispatch_workgroups(workgroups.x, workgroups.y, workgroups.z);
    }

    let count_staging = device::readback_buffer(device, 8, "warpstate literal count staging");
    encoder.copy_buffer_to_buffer(&count_buf, 0, &count_staging, 0, 8);
    encoder.copy_buffer_to_buffer(&candidate_count_buf, 0, &candidate_count_staging, 0, 8);
    let match_staging =
        device::readback_buffer(device, match_buf_size, "warpstate literal match staging");
    encoder.copy_buffer_to_buffer(&match_buf, 0, &match_staging, 0, match_buf_size);
    queue.submit(Some(encoder.finish()));

    let result: Result<Vec<Match>> = async {
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

        if candidate_overflow {
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
                let end_offset =
                    base_offset_u32
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

        readback::read_matches(
            device,
            queue,
            &count_staging,
            &match_staging,
            Some(&literal.pattern_ids),
            base_offset,
            max_matches,
            data.len(),
        )
        .await
    }
    .await;

    candidate_count_staging.destroy();
    count_staging.destroy();
    match_staging.destroy();
    buffer_pool.return_buffer(candidate_buf, candidate_usage);
    buffer_pool.return_buffer(candidate_count_buf, candidate_count_usage);
    buffer_pool.return_buffer(match_buf, match_usage);
    buffer_pool.return_buffer(count_buf, count_usage);
    buffer_pool.return_buffer(prefilter_uniform_buf, uniform_usage);
    buffer_pool.return_buffer(verify_uniform_buf, uniform_usage);
    buffer_pool.return_buffer(input_buf, input_usage);

    result
}

/// Workgroup dispatch dimensions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WorkgroupDims {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Compute the workgroup dimensions needed for an input.
/// Uses 2D dispatch when the 1D workgroup count would exceed the device limit.
pub(crate) fn compute_workgroups(device: &wgpu::Device, input_len: u32) -> Result<WorkgroupDims> {
    let raw = input_len.div_ceil(shader::WORKGROUP_SIZE);
    let max = device.limits().max_compute_workgroups_per_dimension;
    // Shaders support 2D dispatch via gid.y to handle larger inputs.
    let max_x = max.min(65_535);
    if raw > max_x * max {
        return Err(Error::GpuDeviceError {
            reason: format!(
                "input requires {raw} workgroups, exceeding 2D device limit {max_x}×{max}. \
                 Fix: reduce chunk size or max input size."
            ),
        });
    }
    let x = raw.min(max_x);
    let y = raw.div_ceil(max_x);
    Ok(WorkgroupDims { x, y, z: 1 })
}

pub(crate) async fn scan_regex_chunk(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _buffer_pool: &GpuBufferPool,
    pattern_set: &PatternSet,
    regex: &RegexGpu,
    data: &[u8],
    base_offset: usize,
    max_regex_input_size: usize,
) -> Result<Vec<Match>> {
    if data.len() > max_regex_input_size {
        return Err(Error::InputTooLarge {
            bytes: data.len(),
            max_bytes: max_regex_input_size,
        });
    }
    if regex.small_dfas.len() != pattern_set.ir().regex_dfas().len() {
        return Err(Error::GpuDeviceError {
            reason: format!(
                "regex ensemble DFA count {} does not match compiled regex DFA count {}. Fix: rebuild the GPU matcher after changing the pattern set.",
                regex.small_dfas.len(),
                pattern_set.ir().regex_dfas().len(),
            ),
        });
    }

    let matched = regex.matcher.scan_async(data, &regex.small_dfas).await?;
    let mut matches = Vec::new();
    for (matched_regex, dfa) in matched
        .into_iter()
        .zip(pattern_set.ir().regex_dfas().iter())
    {
        if !matched_regex {
            continue;
        }
        dfa.scan_native_with(data, &mut |mut mat| {
            let Ok(start) = u32::try_from(base_offset.saturating_add(mat.start as usize)) else {
                return false;
            };
            let Ok(end) = u32::try_from(base_offset.saturating_add(mat.end as usize)) else {
                return false;
            };
            mat.start = start;
            mat.end = end;
            matches.push(mat);
            true
        })?;
    }
    Ok(matches)
}
