use crate::algebraic::{
    shader, HYBRID_DEAD_STATE, HYBRID_ESCAPE_STATE, HYBRID_GPU_STATES, MAX_ALGEBRAIC_STATES,
    MAX_MATCHES, MAX_SCAN_ROUNDS,
};
use crate::dfa::RegexDFA;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub(crate) struct StaticResources {
    pub(crate) transition_table: wgpu::Buffer,
    pub(crate) match_list_pointers: wgpu::Buffer,
    pub(crate) match_lists: wgpu::Buffer,
    pub(crate) pattern_lengths: wgpu::Buffer,
}

impl StaticResources {
    pub(crate) fn new(device: &wgpu::Device, dfa: &RegexDFA, pattern_lengths: &[u32]) -> Self {
        Self {
            transition_table: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic transition table"),
                contents: bytemuck::cast_slice(&dfa.transition_table),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            match_list_pointers: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic match list pointers"),
                contents: bytemuck::cast_slice(&dfa.match_list_pointers),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            match_lists: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic match lists"),
                contents: bytemuck::cast_slice(&dfa.match_lists),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            pattern_lengths: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic pattern lengths"),
                contents: bytemuck::cast_slice(pattern_lengths),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        }
    }

    pub(crate) fn new_hybrid(device: &wgpu::Device, dfa: &RegexDFA) -> Self {
        let class_count = dfa.class_count as usize;
        let mut transition_table =
            vec![HYBRID_DEAD_STATE; HYBRID_GPU_STATES as usize * class_count];
        let mut match_list_pointers = vec![0u32; HYBRID_GPU_STATES as usize];
        let mut match_lists = vec![0u32];

        for state in 0..MAX_ALGEBRAIC_STATES as usize {
            for class_id in 0..class_count {
                let next = dfa.transition_for_class(state as u32, class_id);
                let next_index = (next & 0x3FFF_FFFF) as usize;
                let mapped = if RegexDFA::is_dead_state(next) {
                    HYBRID_DEAD_STATE
                } else if next_index < MAX_ALGEBRAIC_STATES as usize {
                    next_index as u32
                } else {
                    HYBRID_ESCAPE_STATE
                };
                transition_table[state * class_count + class_id] = mapped;
            }

            let ptr = dfa.match_list_pointers.get(state).copied().unwrap_or(0) as usize;
            let qty = dfa.match_lists.get(ptr).copied().unwrap_or(0) as usize;
            if qty > 0 && ptr + 1 + qty <= dfa.match_lists.len() {
                match_list_pointers[state] = match_lists.len() as u32;
                match_lists.push(qty as u32);
                match_lists.extend_from_slice(&dfa.match_lists[ptr + 1..ptr + 1 + qty]);
            }
        }

        for sentinel in [HYBRID_ESCAPE_STATE as usize, HYBRID_DEAD_STATE as usize] {
            for class_id in 0..class_count {
                transition_table[sentinel * class_count + class_id] = sentinel as u32;
            }
        }

        Self {
            transition_table: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic transition table"),
                contents: bytemuck::cast_slice(&transition_table),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            match_list_pointers: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic match list pointers"),
                contents: bytemuck::cast_slice(&match_list_pointers),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            match_lists: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic match lists"),
                contents: bytemuck::cast_slice(&match_lists),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            pattern_lengths: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("algebraic pattern lengths"),
                contents: bytemuck::cast_slice(dfa.pattern_lengths()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        }
    }
}

pub(crate) struct Pipelines {
    pub(crate) map_pipeline: wgpu::ComputePipeline,
    pub(crate) scan_pipeline: wgpu::ComputePipeline,
    pub(crate) extract_pipeline: wgpu::ComputePipeline,
    pub(crate) map_layout: wgpu::BindGroupLayout,
    pub(crate) scan_layout: wgpu::BindGroupLayout,
    pub(crate) extract_layout: wgpu::BindGroupLayout,
}

impl Pipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let map_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("algebraic map shader"),
            source: wgpu::ShaderSource::Wgsl(shader::generate_map_shader().into()),
        });
        let scan_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("algebraic scan shader"),
            source: wgpu::ShaderSource::Wgsl(shader::generate_scan_shader().into()),
        });
        let extract_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("algebraic extract shader"),
            source: wgpu::ShaderSource::Wgsl(shader::generate_extract_shader().into()),
        });

        let map_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("algebraic map layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
        let scan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("algebraic scan layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });
        let extract_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("algebraic extract layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                uniform_entry(6),
            ],
        });

        let map_pipeline = create_compute_pipeline(
            device,
            &map_module,
            "map_main",
            Some(&map_layout),
            "algebraic map pipeline",
        );
        let scan_pipeline = create_compute_pipeline(
            device,
            &scan_module,
            "scan_main",
            Some(&scan_layout),
            "algebraic scan pipeline",
        );
        let extract_pipeline = create_compute_pipeline(
            device,
            &extract_module,
            "extract_main",
            Some(&extract_layout),
            "algebraic extract pipeline",
        );

        Self {
            map_pipeline,
            scan_pipeline,
            extract_pipeline,
            map_layout,
            scan_layout,
            extract_layout,
        }
    }
}

#[derive(Debug)]
pub(crate) struct AlgebraicState {
    pub(crate) input_buf: wgpu::Buffer,
    pub(crate) func_a: wgpu::Buffer,
    pub(crate) func_b: wgpu::Buffer,
    pub(crate) func_staging: wgpu::Buffer,
    pub(crate) map_uniform_buf: wgpu::Buffer,
    pub(crate) scan_uniform_bufs: Vec<wgpu::Buffer>,
    pub(crate) extract_uniform_buf: wgpu::Buffer,
    pub(crate) match_buf: wgpu::Buffer,
    pub(crate) count_buf: wgpu::Buffer,
    pub(crate) count_staging: wgpu::Buffer,
    pub(crate) match_staging: wgpu::Buffer,
    pub(crate) tail_staging: wgpu::Buffer,
    pub(crate) map_bind_group: wgpu::BindGroup,
    pub(crate) scan_bind_groups_a_to_b: Vec<wgpu::BindGroup>,
    pub(crate) scan_bind_groups_b_to_a: Vec<wgpu::BindGroup>,
    pub(crate) extract_bind_group_a: wgpu::BindGroup,
    pub(crate) extract_bind_group_b: wgpu::BindGroup,
}

impl AlgebraicState {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(
        device: &wgpu::Device,
        resources: &StaticResources,
        state_count: u32,
        block_size: usize,
        map_layout: &wgpu::BindGroupLayout,
        scan_layout: &wgpu::BindGroupLayout,
        extract_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let padded_input_size = block_size.next_multiple_of(4) as u64;
        let func_table_size =
            (block_size * state_count as usize * std::mem::size_of::<u32>()) as u64;
        let match_buf_size = u64::from(MAX_MATCHES) * 16;
        let tail_size =
            (u64::from(state_count) * std::mem::size_of::<u32>() as u64).next_multiple_of(8);

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic input"),
            size: padded_input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let func_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic func_a"),
            size: func_table_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let func_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic func_b"),
            size: func_table_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let func_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic func staging"),
            size: func_table_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let map_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic map uniforms"),
            size: std::mem::size_of::<shader::MapUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let extract_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic extract uniforms"),
            size: std::mem::size_of::<shader::ExtractUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic count"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let match_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic match output"),
            size: match_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic count staging"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic match staging"),
            size: match_buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let tail_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("algebraic tail staging"),
            size: tail_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("algebraic map bind group"),
            layout: map_layout,
            entries: &[
                bind_buffer(0, &input_buf),
                bind_buffer(1, &resources.transition_table),
                bind_buffer(2, &func_a),
                bind_buffer(3, &map_uniform_buf),
            ],
        });

        let mut scan_uniform_bufs = Vec::with_capacity(MAX_SCAN_ROUNDS);
        let mut scan_bind_groups_a_to_b = Vec::with_capacity(MAX_SCAN_ROUNDS);
        let mut scan_bind_groups_b_to_a = Vec::with_capacity(MAX_SCAN_ROUNDS);
        for _ in 0..MAX_SCAN_ROUNDS {
            let scan_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("algebraic scan uniforms"),
                size: std::mem::size_of::<shader::ScanUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("algebraic scan a_to_b"),
                layout: scan_layout,
                entries: &[
                    bind_buffer(0, &func_a),
                    bind_buffer(1, &func_b),
                    bind_buffer(2, &scan_uniform_buf),
                ],
            });
            let bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("algebraic scan b_to_a"),
                layout: scan_layout,
                entries: &[
                    bind_buffer(0, &func_b),
                    bind_buffer(1, &func_a),
                    bind_buffer(2, &scan_uniform_buf),
                ],
            });
            scan_uniform_bufs.push(scan_uniform_buf);
            scan_bind_groups_a_to_b.push(bind_group_a_to_b);
            scan_bind_groups_b_to_a.push(bind_group_b_to_a);
        }

        let extract_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("algebraic extract a"),
            layout: extract_layout,
            entries: &[
                bind_buffer(0, &func_a),
                bind_buffer(1, &resources.match_list_pointers),
                bind_buffer(2, &resources.match_lists),
                bind_buffer(3, &resources.pattern_lengths),
                bind_buffer(4, &match_buf),
                bind_buffer(5, &count_buf),
                bind_buffer(6, &extract_uniform_buf),
            ],
        });
        let extract_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("algebraic extract b"),
            layout: extract_layout,
            entries: &[
                bind_buffer(0, &func_b),
                bind_buffer(1, &resources.match_list_pointers),
                bind_buffer(2, &resources.match_lists),
                bind_buffer(3, &resources.pattern_lengths),
                bind_buffer(4, &match_buf),
                bind_buffer(5, &count_buf),
                bind_buffer(6, &extract_uniform_buf),
            ],
        });

        Self {
            input_buf,
            func_a,
            func_b,
            func_staging,
            map_uniform_buf,
            scan_uniform_bufs,
            extract_uniform_buf,
            match_buf,
            count_buf,
            count_staging,
            match_staging,
            tail_staging,
            map_bind_group,
            scan_bind_groups_a_to_b,
            scan_bind_groups_b_to_a,
            extract_bind_group_a,
            extract_bind_group_b,
        }
    }
}

pub(crate) fn create_compute_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    bind_group_layout: Option<&wgpu::BindGroupLayout>,
    label: &str,
) -> wgpu::ComputePipeline {
    let bind_group_layouts = bind_group_layout.map_or_else(Vec::new, |layout| vec![layout]);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bind_group_layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
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

pub(crate) fn bind_buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
