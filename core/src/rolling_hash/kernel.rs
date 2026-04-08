use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

use crate::config::DEFAULT_MAX_MATCHES;
use crate::error::{Error, Result};
use crate::rolling_hash::{
    HashPipelineState, HashTableEntry, HashUniforms, LengthGroup, PatternEntry, RollingHashMatcher,
};
use crate::shader_hash;

pub async fn acquire_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok_or(Error::NoGpuAdapter)?;

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rolling-hash device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            },
            None,
        )
        .await
        .map_err(|err| Error::GpuDeviceError {
            reason: err.to_string(),
        })
}

pub fn compile_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader_source = shader_hash::generate_rolling_hash_shader();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rolling-hash shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rolling-hash bind group layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            storage_entry(5, false),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rolling-hash pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rolling-hash pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

pub fn build_length_groups(device: &wgpu::Device, patterns: &[&[u8]]) -> Result<Vec<LengthGroup>> {
    let grouped = RollingHashMatcher::group_patterns_by_length(patterns)?;
    let total_patterns =
        u32::try_from(patterns.len()).map_err(|_| Error::PatternCompilationFailed {
            reason: "pattern count exceeds 32-bit address space".to_string(),
        })?;
    let mut groups = Vec::with_capacity(grouped.len());

    for group in grouped {
        let first_length = group.first().map(|(_, bytes)| bytes.len()).ok_or_else(|| {
            Error::PatternCompilationFailed {
                reason: "encountered an empty length group during compilation".to_string(),
            }
        })?;
        let length = u32::try_from(first_length).map_err(|_| Error::PatternCompilationFailed {
            reason: "pattern length exceeds 32-bit address space".to_string(),
        })?;

        let entries: Vec<PatternEntry> = group
            .into_iter()
            .map(|(pattern_id, raw_bytes)| PatternEntry {
                pattern_id,
                hash: RollingHashMatcher::compute_fnv1a(&raw_bytes),
                raw_bytes,
            })
            .collect();

        let hash_table_size = next_hash_table_size(entries.len())?;
        let hash_table = build_hash_table(&entries, hash_table_size)?;
        let pattern_offsets = build_pattern_offsets(&entries, total_patterns)?;
        let pattern_bytes = pack_group_pattern_bytes(&entries)?;

        let hash_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rolling-hash hash table"),
            contents: bytemuck::cast_slice(&hash_table),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pattern_bytes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rolling-hash pattern bytes"),
            contents: &pattern_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pattern_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rolling-hash pattern offsets"),
            contents: bytemuck::cast_slice(&pattern_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        groups.push(LengthGroup {
            length,
            hash_table_size,
            hash_table_buf,
            pattern_bytes_buf,
            pattern_offsets_buf,
        });
    }

    Ok(groups)
}

fn next_hash_table_size(entry_count: usize) -> Result<u32> {
    let desired = entry_count
        .checked_mul(2)
        .ok_or_else(|| Error::PatternCompilationFailed {
            reason: "hash table size overflowed while compiling patterns".to_string(),
        })?;
    let size = desired.max(1).checked_next_power_of_two().ok_or_else(|| {
        Error::PatternCompilationFailed {
            reason: "hash table size exceeds supported power-of-two range".to_string(),
        }
    })?;
    u32::try_from(size).map_err(|_| Error::PatternCompilationFailed {
        reason: "hash table size exceeds 32-bit address space".to_string(),
    })
}

fn build_hash_table(entries: &[PatternEntry], table_size: u32) -> Result<Vec<HashTableEntry>> {
    let mut table = vec![HashTableEntry::zeroed(); table_size as usize];
    let mask = table_size - 1;

    for entry in entries {
        let mut inserted = false;
        for probe in 0..table_size {
            let slot = (entry.hash.wrapping_add(probe) & mask) as usize;
            if table[slot].occupied == 0 {
                table[slot] = HashTableEntry {
                    occupied: 1,
                    hash: entry.hash,
                    pattern_id: entry.pattern_id,
                    padding: 0,
                };
                inserted = true;
                break;
            }
        }

        if !inserted {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "failed to insert pattern {} into rolling-hash table",
                    entry.pattern_id
                ),
            });
        }
    }

    Ok(table)
}

fn build_pattern_offsets(entries: &[PatternEntry], total_patterns: u32) -> Result<Vec<u32>> {
    let mut offsets = vec![0u32; total_patterns as usize];
    let mut cursor = 0u32;

    for entry in entries {
        let length =
            u32::try_from(entry.raw_bytes.len()).map_err(|_| Error::PatternCompilationFailed {
                reason: format!(
                    "pattern {} length exceeded 32-bit address space",
                    entry.pattern_id
                ),
            })?;
        offsets[entry.pattern_id as usize] = cursor;
        cursor = cursor
            .checked_add(length)
            .ok_or_else(|| Error::PatternCompilationFailed {
                reason: "packed pattern bytes exceeded 32-bit address space".to_string(),
            })?;
    }

    Ok(offsets)
}

fn pack_group_pattern_bytes(entries: &[PatternEntry]) -> Result<Vec<u8>> {
    let total_bytes = entries.iter().try_fold(0usize, |acc, entry| {
        acc.checked_add(entry.raw_bytes.len())
            .ok_or_else(|| Error::PatternCompilationFailed {
                reason: "packed pattern bytes overflowed usize".to_string(),
            })
    })?;
    let padded_len = total_bytes.next_multiple_of(4);
    let mut packed = vec![0u8; padded_len];
    let mut cursor = 0usize;

    for entry in entries {
        let end = cursor + entry.raw_bytes.len();
        packed[cursor..end].copy_from_slice(&entry.raw_bytes);
        cursor = end;
    }

    Ok(packed)
}

pub fn allocate_buffers(device: &wgpu::Device, max_input_size: usize) -> HashPipelineState {
    let padded_len = max_input_size.next_multiple_of(4);
    let match_buf_size = u64::from(DEFAULT_MAX_MATCHES) * 16;

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rolling-hash input"),
        size: padded_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let match_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rolling-hash match buffer"),
        size: match_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rolling-hash count buffer"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rolling-hash uniform buffer"),
        size: std::mem::size_of::<HashUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rolling-hash count staging"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rolling-hash match staging"),
        size: match_buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    HashPipelineState {
        input_buf,
        match_buf,
        count_buf,
        uniform_buf,
        count_staging,
        match_staging,
    }
}

pub fn upload_input(queue: &wgpu::Queue, input_buf: &wgpu::Buffer, data: &[u8]) {
    let aligned_len = data.len() & !3;
    if aligned_len > 0 {
        queue.write_buffer(input_buf, 0, &data[..aligned_len]);
    }

    let remainder = data.len() - aligned_len;
    if remainder > 0 {
        let mut tail = [0u8; 4];
        tail[..remainder].copy_from_slice(&data[aligned_len..]);
        queue.write_buffer(input_buf, aligned_len as u64, &tail);
    }
}

pub fn reset_count_and_uniforms(
    queue: &wgpu::Queue,
    state: &HashPipelineState,
    input_len: u32,
    pattern_length: u32,
    hash_table_size: u32,
) {
    queue.write_buffer(&state.count_buf, 0, bytemuck::cast_slice(&[0u32, 0u32]));

    let uniforms = HashUniforms {
        input_len,
        pattern_length,
        hash_table_size,
        max_matches: DEFAULT_MAX_MATCHES,
    };
    queue.write_buffer(&state.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
}

pub fn assemble_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    group: &LengthGroup,
    state: &HashPipelineState,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rolling-hash bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: group.hash_table_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: group.pattern_bytes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: group.pattern_offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state.match_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state.count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: state.uniform_buf.as_entire_binding(),
            },
        ],
    })
}

pub fn dispatch_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    state: &HashPipelineState,
    input_len: u32,
) {
    let total_workgroups = input_len.div_ceil(shader_hash::WORKGROUP_SIZE);
    let workgroups_x = total_workgroups.min(65_535);
    let workgroups_y = total_workgroups.div_ceil(65_535);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rolling-hash encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    let match_buf_size = u64::from(DEFAULT_MAX_MATCHES) * 16;
    encoder.copy_buffer_to_buffer(&state.count_buf, 0, &state.count_staging, 0, 8);
    encoder.copy_buffer_to_buffer(&state.match_buf, 0, &state.match_staging, 0, match_buf_size);
    queue.submit(Some(encoder.finish()));
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
