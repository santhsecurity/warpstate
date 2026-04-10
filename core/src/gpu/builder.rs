use wgpu::util::DeviceExt;

use crate::error::{Error, Result};
use crate::gpu::device::{storage_entry, uniform_entry};
use crate::gpu::dispatch::{LiteralGpu, RegexGpu, SpecializedRegexGpu};
use crate::pattern::{CompiledPatternKind, PatternSet};
use crate::shader;

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

    let pattern_count = patterns.ir().offsets.len().try_into().map_err(|_| {
        Error::PatternSetTooLarge {
            patterns: patterns.len(),
            bytes: patterns.ir().packed_bytes.len(),
            max_bytes: u32::MAX as usize,
        }
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
    patterns: &PatternSet,
) -> Result<Option<RegexGpu>> {
    if patterns.ir().regex_dfas.is_empty() {
        return Ok(None);
    }
    let dfa = &patterns.ir().regex_dfas[0];

    let max_storage = device.limits().max_storage_buffer_binding_size as usize;
    let transitions_bytes = dfa
        .transition_table()
        .len()
        .saturating_mul(std::mem::size_of::<u32>());
    if transitions_bytes > max_storage {
        return Err(Error::PatternCompilationFailed {
            reason: format!(
                "regex DFA transition table is {transitions_bytes} bytes, exceeding GPU max_storage_buffer_binding_size ({max_storage} bytes). Fix: reduce regex complexity or use CPU scanning."
            ),
        });
    }

    let transitions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate regex transitions"),
        contents: bytemuck::cast_slice(dfa.transition_table()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let match_list_pointers_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate regex match list pointers"),
        contents: bytemuck::cast_slice(dfa.match_list_pointers()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let match_lists_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate regex match lists"),
        contents: bytemuck::cast_slice(dfa.match_lists()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pattern_lengths_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warpstate regex pattern lengths"),
        contents: bytemuck::cast_slice(dfa.pattern_lengths()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("warpstate regex shader"),
        source: wgpu::ShaderSource::Wgsl(shader::generate_regex_dfa_shader().into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warpstate regex layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            storage_entry(5, false),
            uniform_entry(6),
            storage_entry(7, true),
        ],
    });
    let pipeline = build_pipeline(device, &layout, &shader_module, "warpstate regex pipeline");

    let original_ids: Vec<usize> = dfa.native_original_ids().to_vec();

    Ok(Some(RegexGpu {
        pipeline,
        layout,
        transitions_buf,
        match_list_pointers_buf,
        match_lists_buf,
        pattern_lengths_buf,
        start_state: dfa.start_state,
        class_count: dfa.class_count,
        eoi_class: dfa.eoi_class,
        byte_classes: dfa.byte_classes,
        pattern_ids: original_ids,
    }))
}

/// Build a specialized regex GPU pipeline with DFA transitions as WGSL constants.
/// Returns `None` if the DFA is too large for constant embedding or if no regex patterns exist.
pub(crate) fn build_specialized_regex_gpu(
    device: &wgpu::Device,
    patterns: &PatternSet,
) -> Result<Option<SpecializedRegexGpu>> {
    if patterns.ir().regex_dfas.is_empty() {
        return Ok(None);
    }
    let dfa = &patterns.ir().regex_dfas[0];

    let shader_source = match shader::generate_specialized_dfa_shader(
        dfa.transition_table(),
        dfa.match_list_pointers(),
        dfa.match_lists(),
        dfa.pattern_lengths(),
        &dfa.byte_classes,
        dfa.start_state,
        dfa.class_count,
        dfa.eoi_class,
    ) {
        Some(source) => source,
        None => return Ok(None),
    };

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("warpstate specialized regex shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Specialized shader uses only 4 bindings:
    // 0: input_data (storage, read)
    // 1: match_output (storage, read_write)
    // 2: match_count (storage, read_write)
    // 3: uniforms (uniform)
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warpstate specialized regex layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let pipeline = build_pipeline(
        device,
        &layout,
        &shader_module,
        "warpstate specialized regex pipeline",
    );

    let original_ids: Vec<usize> = dfa.native_original_ids().to_vec();

    Ok(Some(SpecializedRegexGpu {
        pipeline,
        layout,
        pattern_ids: original_ids,
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
