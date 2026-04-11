use wgpu::util::DeviceExt;

use crate::error::{Error, Result};
use crate::gpu::device::{storage_entry, uniform_entry};
use crate::gpu::dispatch::{LiteralGpu, RegexGpu};
use crate::gpu::ensemble::{EnsembleRegexMatcher, SmallDfa};
use crate::gpu::shader;
use crate::pattern::{CompiledPatternKind, PatternSet};

pub(crate) fn build_literal_gpu(
    device: &wgpu::Device,
    patterns: &PatternSet,
) -> Result<Option<LiteralGpu>> {
    if patterns.ir().offsets.is_empty() {
        return Ok(None);
    }

    let packed_bytes = crate::gpu::device::pad_to_u32(&patterns.ir().packed_bytes);
    let packed_offsets: Vec<u32> = patterns
        .ir()
        .offsets
        .iter()
        .flat_map(|&(offset, len)| [offset, len])
        .collect();
    let pattern_ids: Vec<usize> = patterns
        .ir()
        .matchers
        .iter()
        .filter_map(|matcher| match matcher.kind {
            CompiledPatternKind::Literal { .. } => Some(matcher.id),
            CompiledPatternKind::Regex => None,
        })
        .collect();

    let pattern_count =
        patterns
            .ir()
            .offsets
            .len()
            .try_into()
            .map_err(|_| Error::PatternSetTooLarge {
                patterns: patterns.len(),
                bytes: patterns.ir().packed_bytes.len(),
                max_bytes: u32::MAX as usize,
            })?;

    let pattern_bytes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate literal pattern bytes"),
        contents: crate::gpu::device::packed_u32_as_bytes(&packed_bytes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pattern_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate literal pattern offsets"),
        contents: bytemuck::cast_slice(&packed_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let prefilter_prefix_meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate literal prefilter prefix metadata"),
        contents: bytemuck::cast_slice(&patterns.ir().literal_prefilter_table.prefix_meta),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let prefilter_bucket_ranges_buf =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warpstate literal prefilter bucket ranges"),
            contents: bytemuck::cast_slice(&patterns.ir().literal_prefilter_table.bucket_ranges),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let prefilter_entries_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate literal prefilter entries"),
        contents: bytemuck::cast_slice(&patterns.ir().literal_prefilter_table.entries),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let prefilter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("warpstate literal prefilter shader"),
        source: wgpu::ShaderSource::Wgsl(shader::generate_literal_prefilter_shader().into()),
    });
    let verify_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("warpstate literal verify shader"),
        source: wgpu::ShaderSource::Wgsl(shader::generate_literal_verify_shader().into()),
    });
    let prefilter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warpstate literal prefilter layout"),
        entries: &[
            storage_entry(0, true),  // input_data
            storage_entry(1, true),  // prefix_meta
            storage_entry(2, true),  // bucket_ranges
            storage_entry(3, true),  // bucket_entries
            storage_entry(4, true),  // pattern_offsets
            storage_entry(5, false), // candidates (read_write)
            storage_entry(6, false), // candidate_count (atomic, read_write)
            uniform_entry(7),        // uniforms
        ],
    });
    let verify_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warpstate literal verify layout"),
        entries: &[
            storage_entry(0, true),  // input_data
            storage_entry(1, true),  // pattern_bytes
            storage_entry(2, true),  // pattern_offsets
            storage_entry(3, true),  // candidates (read)
            storage_entry(4, true),  // candidate_count (read)
            storage_entry(5, false), // match_output (read_write)
            storage_entry(6, false), // match_count (atomic, read_write)
            uniform_entry(7),        // uniforms
        ],
    });
    let prefilter_pipeline = build_pipeline(
        device,
        &prefilter_layout,
        &prefilter_shader,
        "warpstate literal prefilter pipeline",
    );
    let verify_pipeline = build_pipeline(
        device,
        &verify_layout,
        &verify_shader,
        "warpstate literal verify pipeline",
    );

    Ok(Some(LiteralGpu {
        prefilter_pipeline,
        verify_pipeline,
        prefilter_layout,
        verify_layout,
        pattern_bytes_buf,
        pattern_offsets_buf,
        prefilter_prefix_meta_buf,
        prefilter_bucket_ranges_buf,
        prefilter_entries_buf,
        pattern_count,
        pattern_ids,
        hash_window_len: patterns.ir().hash_window_len,
    }))
}

pub(crate) fn build_regex_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    patterns: &PatternSet,
) -> Result<Option<RegexGpu>> {
    if patterns.ir().regex_dfas.is_empty() {
        return Ok(None);
    }
    let small_dfas = patterns
        .ir()
        .regex_dfas()
        .iter()
        .map(SmallDfa::from_regex_dfa)
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(RegexGpu {
        matcher: EnsembleRegexMatcher::new(device.clone(), queue.clone()),
        small_dfas,
    }))
}

pub(crate) fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    label: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}
