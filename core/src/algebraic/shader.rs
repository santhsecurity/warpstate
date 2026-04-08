//! WGSL shader generation for algebraic DFA prefix-scan execution.

/// Workgroup size for the algebraic compute shaders.
pub const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MapUniforms {
    pub(crate) input_len: u32,
    pub(crate) state_count: u32,
    pub(crate) class_count: u32,
    pub(crate) _padding0: u32,
    pub(crate) byte_classes: [[u32; 4]; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ScanUniforms {
    pub(crate) input_len: u32,
    pub(crate) state_count: u32,
    pub(crate) stride: u32,
    pub(crate) padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ExtractUniforms {
    pub(crate) input_len: u32,
    pub(crate) state_count: u32,
    pub(crate) carry_state: u32,
    pub(crate) max_matches: u32,
    pub(crate) block_offset: u32,
    pub(crate) padding0: u32,
    pub(crate) padding1: u32,
    pub(crate) padding2: u32,
}

/// Generate the algebraic map shader.
pub fn generate_map_shader() -> String {
    format!(
        r#"struct MapUniforms {{
    input_len: u32,
    state_count: u32,
    class_count: u32,
    _padding0: u32,
    byte_classes: array<vec4<u32>, 64>,
}}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read> transition_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> func_table: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: MapUniforms;

const MASK_STATE: u32 = 0x3FFFFFFFu;

@compute @workgroup_size({WORKGROUP_SIZE})
fn map_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pos = gid.x;
    if (pos >= uniforms.input_len) {{
        return;
    }}

    let word_idx = pos >> 2u;
    let byte_offset = (pos & 3u) * 8u;
    let byte_val = (input_data[word_idx] >> byte_offset) & 0xFFu;
    let class_id = uniforms.byte_classes[byte_val >> 2u][byte_val & 3u];
    let base = pos * uniforms.state_count;

    for (var state = 0u; state < uniforms.state_count; state = state + 1u) {{
        let next = transition_table[state * uniforms.class_count + class_id];
        func_table[base + state] = next & MASK_STATE;
    }}
}}"#
    )
}

/// Generate the algebraic Hillis-Steele scan shader.
pub fn generate_scan_shader() -> String {
    format!(
        r#"struct ScanUniforms {{
    input_len: u32,
    state_count: u32,
    stride: u32,
    _padding0: u32,
}}

@group(0) @binding(0) var<storage, read> func_table_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> func_table_out: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: ScanUniforms;

@compute @workgroup_size({WORKGROUP_SIZE})
fn scan_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pos = gid.x;
    if (pos >= uniforms.input_len) {{
        return;
    }}

    let base = pos * uniforms.state_count;
    if (pos < uniforms.stride) {{
        for (var state = 0u; state < uniforms.state_count; state = state + 1u) {{
            func_table_out[base + state] = func_table_in[base + state];
        }}
        return;
    }}

    let src_base = (pos - uniforms.stride) * uniforms.state_count;
    for (var state = 0u; state < uniforms.state_count; state = state + 1u) {{
        let intermediate = func_table_in[src_base + state];
        func_table_out[base + state] = func_table_in[base + intermediate];
    }}
}}"#
    )
}

/// Generate the algebraic extract shader.
pub fn generate_extract_shader() -> String {
    format!(
        r#"struct ExtractUniforms {{
    input_len: u32,
    state_count: u32,
    carry_state: u32,
    max_matches: u32,
    block_offset: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

@group(0) @binding(0) var<storage, read> func_table: array<u32>;
@group(0) @binding(1) var<storage, read> match_list_pointers: array<u32>;
@group(0) @binding(2) var<storage, read> match_lists: array<u32>;
@group(0) @binding(3) var<storage, read> pattern_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> match_output: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> match_count: array<atomic<u32>, 2>;
@group(0) @binding(6) var<uniform> uniforms: ExtractUniforms;

@compute @workgroup_size({WORKGROUP_SIZE})
fn extract_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pos = gid.x;
    if (pos >= uniforms.input_len) {{
        return;
    }}

    let base = pos * uniforms.state_count;
    let final_state = func_table[base + uniforms.carry_state];
    let match_ptr = match_list_pointers[final_state];
    let qty = match_lists[match_ptr];
    let end = uniforms.block_offset + pos + 1u;

    for (var m = 0u; m < qty; m = m + 1u) {{
        let count = atomicAdd(&match_count[0], 1u);
        if (count < uniforms.max_matches) {{
            let pat_id = match_lists[match_ptr + 1u + m];
            let pat_len = pattern_lengths[pat_id];
            let start = select(0u, end - pat_len, pat_len <= end);
            match_output[count] = vec4<u32>(pat_id, start, end, 0u);
        }} else {{
            atomicStore(&match_count[1], 1u);
        }}
    }}
}}"#
    )
}
