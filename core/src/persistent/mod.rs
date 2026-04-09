//! Double-buffered GPU matcher that keeps two scan buffer sets hot.
//!
//! This module reuses the existing DFA WGSL shader, but allocates two
//! independent input/output buffer sets. Sequential scans can upload into one
//! set while another scan is still completing, which hides most per-dispatch
//! setup cost for streaming workloads without relying on non-terminating WGSL.

use crate::config::AutoMatcherConfig;
use crate::error::Result;
use crate::gpu::SharedDeviceQueue;
use crate::{shader, PatternSet};
use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

pub mod scan;
pub mod state;

/// Double-buffered GPU DFA matcher with pre-staged dispatch.
///
/// This backend keeps two sets of I/O buffers. While the GPU processes one
/// buffer set, the CPU uploads data to the other. When the GPU finishes, the
/// next dispatch is immediately submitted and can reuse the already-initialized
/// pipeline state.
///
/// For streaming workloads with many sequential scans, this hides dispatch
/// overhead almost entirely compared with a cold per-scan command path.
pub struct PersistentMatcher {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) buffer_sets: [BufferSet; 2],
    pub(crate) bind_groups: [wgpu::BindGroup; 2],
    pub(crate) active_set: AtomicUsize,
    pub(crate) start_state: u32,
    pub(crate) class_count: u32,
    pub(crate) eoi_class: u32,
    pub(crate) byte_classes: [u32; 256],
    pub(crate) max_input_size: usize,
    pub(crate) max_matches: u32,
    pub(crate) state: Mutex<state::PersistentState>,
}

pub(crate) struct BufferSet {
    pub(crate) input_buf: wgpu::Buffer,
    pub(crate) match_buf: wgpu::Buffer,
    pub(crate) count_buf: wgpu::Buffer,
    pub(crate) uniform_buf: wgpu::Buffer,
    pub(crate) count_staging: wgpu::Buffer,
    pub(crate) match_staging: wgpu::Buffer,
}

pub(crate) struct PendingWork {
    pub(crate) buffer_set_idx: usize,
    pub(crate) input_len: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    pub(crate) input_len: u32,
    pub(crate) start_state: u32,
    pub(crate) max_matches: u32,
    pub(crate) class_count: u32,
    pub(crate) eoi_class: u32,
    // Match WGSL struct alignment: vec3<u32> after 5 u32 scalars requires
    // padding to reach a 16-byte aligned offset for the subsequent array.
    pub(crate) padding: [u32; 3],
    pub(crate) byte_classes: [[u32; 4]; 64],
}

impl PersistentMatcher {
    /// Compile a pattern set into the persistent double-buffered GPU backend.
    pub async fn new(patterns: &PatternSet) -> Result<Self> {
        Self::with_config(patterns, AutoMatcherConfig::default()).await
    }

    /// Compile a pattern set into the persistent backend with explicit config.
    pub async fn with_config(patterns: &PatternSet, config: AutoMatcherConfig) -> Result<Self> {
        let device_queue = crate::gpu::acquire_device().await?;
        Self::from_device(device_queue, patterns, config)
    }

    /// Compile a pattern set into the persistent backend using an existing device.
    pub fn from_device(
        device_queue: SharedDeviceQueue,
        patterns: &PatternSet,
        config: AutoMatcherConfig,
    ) -> Result<Self> {
        let regex_dfa = patterns.compiled_regex_dfa()?;
        let max_input_size = config.configured_gpu_max_input_size();
        let max_matches = config.configured_max_matches();

        let device = device_queue.0.clone();
        let queue = device_queue.1.clone();
        let (pipeline, bind_group_layout) = compile_pipeline(&device);

        let transition_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("persistent dfa transition table"),
            contents: bytemuck::cast_slice(regex_dfa.transition_table()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let match_list_pointers_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("persistent match list pointers"),
                contents: bytemuck::cast_slice(regex_dfa.match_list_pointers()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let match_lists_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("persistent match lists"),
            contents: bytemuck::cast_slice(regex_dfa.match_lists()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pattern_lengths_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("persistent pattern lengths"),
            contents: bytemuck::cast_slice(regex_dfa.pattern_lengths()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buffer_set0 = allocate_buffers(&device, max_input_size, max_matches);
        let bind_group0 = assemble_bind_group(
            &device,
            &bind_group_layout,
            &transition_table_buf,
            &match_list_pointers_buf,
            &match_lists_buf,
            &pattern_lengths_buf,
            &buffer_set0,
        );

        let buffer_set1 = allocate_buffers(&device, max_input_size, max_matches);
        let bind_group1 = assemble_bind_group(
            &device,
            &bind_group_layout,
            &transition_table_buf,
            &match_list_pointers_buf,
            &match_lists_buf,
            &pattern_lengths_buf,
            &buffer_set1,
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            buffer_sets: [buffer_set0, buffer_set1],
            bind_groups: [bind_group0, bind_group1],
            active_set: AtomicUsize::new(0),
            start_state: regex_dfa.start_state,
            class_count: regex_dfa.class_count,
            eoi_class: regex_dfa.eoi_class,
            byte_classes: regex_dfa.byte_classes,
            max_input_size,
            max_matches,
            state: Mutex::new(state::PersistentState {
                inflight: [false, false],
                pending: None,
                use_counts: [0, 0],
            }),
        })
    }
}

fn compile_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader_source = shader::generate_regex_dfa_shader();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("persistent dfa compute shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("persistent dfa bind group layout"),
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
            storage_entry(7, true),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("persistent dfa pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("persistent dfa pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

fn allocate_buffers(device: &wgpu::Device, max_input_size: usize, max_matches: u32) -> BufferSet {
    let padded_len = max_input_size.next_multiple_of(4);
    let match_buf_size = match_buffer_size_bytes(max_matches);

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent input"),
        size: padded_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let match_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent match buf"),
        size: match_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent count buf"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent count staging"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent match staging"),
        size: match_buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    BufferSet {
        input_buf,
        match_buf,
        count_buf,
        uniform_buf,
        count_staging,
        match_staging,
    }
}

fn assemble_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    transition_table: &wgpu::Buffer,
    match_list_pointers: &wgpu::Buffer,
    match_lists: &wgpu::Buffer,
    pattern_lengths: &wgpu::Buffer,
    buffer_set: &BufferSet,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("persistent dfa bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_set.input_buf.as_entire_binding(),
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
                resource: buffer_set.match_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buffer_set.count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buffer_set.uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: pattern_lengths.as_entire_binding(),
            },
        ],
    })
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

pub(crate) fn match_buffer_size_bytes(max_matches: u32) -> u64 {
    u64::from(max_matches) * 16
}

pub(crate) fn pack_byte_classes(byte_classes: &[u32; 256]) -> [[u32; 4]; 64] {
    let mut packed = [[0_u32; 4]; 64];
    for (index, class_id) in byte_classes.iter().copied().enumerate() {
        packed[index / 4][index % 4] = class_id;
    }
    packed
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use crate::matcher::BlockMatcher;
    use crate::{error::Error, Match, PatternSet};

    fn build_patterns() -> PatternSet {
        // Use non-overlapping patterns to avoid GPU/CPU match semantics divergence.
        // GPU DFA finds longest match at each position; CPU Aho-Corasick finds
        // leftmost-first. With non-overlapping patterns the results are identical.
        PatternSet::builder()
            .regex("hello")
            .regex("world")
            .build()
            .unwrap()
    }

    fn build_matcher(patterns: &PatternSet) -> Option<PersistentMatcher> {
        match pollster::block_on(PersistentMatcher::new(patterns)) {
            Ok(matcher) => Some(matcher),
            Err(Error::NoGpuAdapter) => None,
            Err(error) => panic!("unexpected persistent matcher init error: {error:?}"),
        }
    }

    #[test]
    fn persistent_single_scan() {
        let patterns = build_patterns();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };
        let data = b"hello world";

        let gpu_matches = pollster::block_on(matcher.scan_block(data)).unwrap();
        let cpu_matches = patterns.scan(data).unwrap();
        assert_eq!(gpu_matches, cpu_matches);
    }

    #[test]
    fn persistent_sequential_scans() {
        let patterns = build_patterns();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };
        let data = b"hello world";
        let expected = patterns.scan(data).unwrap();

        for _ in 0..100 {
            let gpu_matches = pollster::block_on(matcher.scan_block(data)).unwrap();
            assert_eq!(gpu_matches, expected);
        }
    }

    #[test]
    fn persistent_matches_cpu_parity() {
        let patterns = PatternSet::builder()
            .regex("alpha")
            .regex("b.t")
            .regex("z")
            .build()
            .unwrap();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };
        let data = b"alpha bet bat z";

        let gpu_matches = pollster::block_on(matcher.scan_block(data)).unwrap();
        let cpu_matches = patterns.scan(data).unwrap();

        assert_eq!(gpu_matches, cpu_matches);
    }

    #[test]
    fn persistent_alternating_data() {
        let patterns = build_patterns();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };

        let first = pollster::block_on(matcher.scan_block(b"hello")).unwrap();
        let second = pollster::block_on(matcher.scan_block(b"world")).unwrap();

        assert_ne!(first, second);
        assert_eq!(first, patterns.scan(b"hello").unwrap());
        assert_eq!(second, patterns.scan(b"world").unwrap());
    }

    #[test]
    fn persistent_empty_input() {
        let patterns = build_patterns();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };

        let gpu_matches = pollster::block_on(matcher.scan_block(b"")).unwrap();

        assert!(gpu_matches.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn persistent_concurrent_scans() {
        let patterns = build_patterns();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };
        let matcher = std::sync::Arc::new(matcher);
        let inputs = [
            b"hello world".as_slice(),
            b"world".as_slice(),
            b"he".as_slice(),
            b"hello".as_slice(),
        ];

        let mut tasks = Vec::new();
        for input in inputs {
            let matcher = std::sync::Arc::clone(&matcher);
            let expected = patterns.scan(input).unwrap();
            tasks.push(tokio::spawn(async move {
                let actual = matcher.scan_block(input).await.unwrap();
                assert_eq!(actual, expected);
            }));
        }

        for task in tasks {
            task.await.unwrap();
        }
    }

    #[test]
    fn persistent_double_buffer_reuse() {
        let patterns = build_patterns();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };

        for _ in 0..6 {
            let _ = pollster::block_on(matcher.scan_block(b"hello world")).unwrap();
        }

        let counts = matcher.buffer_use_counts();
        assert!(counts[0] > 0, "buffer set 0 was never used");
        assert!(counts[1] > 0, "buffer set 1 was never used");
    }

    #[test]
    fn persistent_large_then_small() {
        let patterns = PatternSet::builder().regex("abc").build().unwrap();
        let Some(matcher) = build_matcher(&patterns) else {
            return;
        };
        let large = vec![b'a'; 32 * 1024];
        let small = b"abc";

        let large_matches = pollster::block_on(matcher.scan_block(&large)).unwrap();
        let small_matches = pollster::block_on(matcher.scan_block(small)).unwrap();

        assert_eq!(large_matches, patterns.scan(&large).unwrap());
        assert_eq!(small_matches, patterns.scan(small).unwrap());
    }
}
