//! WGSL compute shader generation for GPU pattern matching.
//!
//! # Shader Transparency
//!
//! - **Algorithms**:
//!   - `literal_prefilter`: Parallel FNV-1a hashing against multiple patterns. Time: O(N).
//!   - `literal_verify`: Exact-match verification for hash collisions. Time: O(N).
//!   - `regex_dfa`: Dense Deterministic Finite Automaton traversal. Time: O(N).
//! - **GPU Memory Requirements**: Proportional to `input_len` + matched pattern offsets. Buffer bindings use `<storage, read>` and `<storage, read_write>`.
//! - **Compilation Strategy**: Shaders are dynamically constructed as WGSL strings at runtime. No external injection is possible because format arguments are restricted to strictly typed internal constants (e.g., `WORKGROUP_SIZE`).

/// Workgroup size for compute shaders.
pub const WORKGROUP_SIZE: u32 = 256;

/// Generate the literal prefilter shader.
/// Outputs a compact list of (position, pattern_index) pairs via atomic append.
/// This eliminates the O(N×P) verify bottleneck by storing WHICH pattern matched.
pub fn generate_literal_prefilter_shader() -> String {
    format!(
        r#"struct Uniforms {{
    input_len: u32,
    pattern_count: u32,
    max_matches: u32,
    hash_window_len: u32,
    _reserved0: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_meta: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bucket_ranges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> bucket_entries: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> pattern_offsets: array<vec2<u32>>;
// Candidate list: append-only array of vec2<u32>(position, pattern_index).
// This replaces the bitmask (1 bit/position) to eliminate O(N×P) verify.
// Buffer is sized to max_matches entries (8 bytes each = pos + pattern_idx).
@group(0) @binding(5) var<storage, read_write> candidates: array<vec2<u32>>;
// Atomic counter for append operations. candidates[atomicAdd(&candidate_count, 1)] = ...
@group(0) @binding(6) var<storage, read_write> candidate_count: array<atomic<u32>, 2>;
@group(0) @binding(7) var<uniform> uniforms: Uniforms;

var<workgroup> byte_shifts: array<u32, 4>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {{
    if local_idx < 4u {{
        byte_shifts[local_idx] = local_idx << 3u;
    }}
    workgroupBarrier();

    let pos = gid.x;
    if pos >= uniforms.input_len {{
        return;
    }}

    var hash = 2166136261u;
    let max_probe_len = min(uniforms.hash_window_len, uniforms.input_len - pos);

    for (var i = 0u; i < max_probe_len; i = i + 1u) {{
        let input_byte_idx = pos + i;
        let input_word_idx = input_byte_idx >> 2u;
        let input_shift = byte_shifts[input_byte_idx & 3u];
        let input_byte = (input_data[input_word_idx] >> input_shift) & 0xFFu;
        hash = (hash ^ input_byte) * 16777619u;

        let prefix_info = prefix_meta[i];
        if prefix_info.z == 0u {{
            continue;
        }}

        let bucket_idx = prefix_info.x + (hash & prefix_info.y);
        let chain = bucket_ranges[bucket_idx];
        for (var entry_idx = 0u; entry_idx < chain.y; entry_idx = entry_idx + 1u) {{
            let entry = bucket_entries[chain.x + entry_idx];
            if entry.x != hash {{
                continue;
            }}

            let pat_len = pattern_offsets[entry.y].y;
            if pos + pat_len > uniforms.input_len {{
                continue;
            }}

            // Emit ALL hash-matching entries at this prefix length as candidates.
            // The verify shader does exact byte comparison on each.
            let idx = atomicAdd(&candidate_count[0], 1u);
            if idx < uniforms.max_matches {{
                candidates[idx] = vec2<u32>(pos, entry.y);
            }} else {{
                atomicStore(&candidate_count[1], 1u);
            }}
        }}
    }}
}}"#
    )
}

/// Generate the literal verification shader.
/// Reads the (position, pattern_index) candidate list and verifies only the named pattern.
/// This changes verify from O(candidates × P) to O(candidates × 1).
pub fn generate_literal_verify_shader() -> String {
    format!(
        r#"struct Uniforms {{
    input_len: u32,
    pattern_count: u32,
    max_matches: u32,
    hash_window_len: u32,
    _reserved0: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read> pattern_bytes: array<u32>;
@group(0) @binding(2) var<storage, read> pattern_offsets: array<vec2<u32>>;
// Candidate list: array of vec2<u32>(position, pattern_index) from prefilter.
// Each entry specifies exactly which pattern to verify at which position.
@group(0) @binding(3) var<storage, read> candidates: array<vec2<u32>>;
// Number of valid entries in candidates list (written by prefilter).
@group(0) @binding(4) var<storage, read> candidate_count: array<u32, 1>;
@group(0) @binding(5) var<storage, read_write> match_output: array<vec3<u32>>;
@group(0) @binding(6) var<storage, read_write> match_count: array<atomic<u32>, 2>;
@group(0) @binding(7) var<uniform> uniforms: Uniforms;

var<workgroup> byte_shifts: array<u32, 4>;
// Tile input into shared memory: each workgroup loads a contiguous chunk.
// 256 threads × 4 bytes = 1024 bytes per tile + max pattern overshoot.
// 4KB shared memory tile covers the workgroup range + longest pattern tail.
var<workgroup> input_tile: array<u32, 1024>;
var<workgroup> tile_base: u32;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {{
    if local_idx < 4u {{
        byte_shifts[local_idx] = local_idx << 3u;
    }}

    // Cooperatively load input tile into shared memory.
    // Each workgroup covers 256 positions. We load 1024 u32s = 4096 bytes
    // to cover the workgroup range (256 bytes) plus pattern overshoot (up to 3840 bytes).
    let wg_start = wg_id.x * {WORKGROUP_SIZE}u;
    if local_idx == 0u {{
        tile_base = wg_start;
    }}
    workgroupBarrier();

    let tile_words = 1024u;
    for (var w = local_idx; w < tile_words; w = w + {WORKGROUP_SIZE}u) {{
        let global_word = (wg_start >> 2u) + w;
        if global_word < arrayLength(&input_data) {{
            input_tile[w] = input_data[global_word];
        }} else {{
            input_tile[w] = 0u;
        }}
    }}
    workgroupBarrier();

    let total_candidates = candidate_count[0];
    let total_threads = num_wg.x * {WORKGROUP_SIZE}u;
    
    // Each thread processes candidates starting at gid.x, striding by total_threads.
    // This distributes the O(candidates) work across all threads in the dispatch.
    var candidate_idx = gid.x;
    while (candidate_idx < total_candidates) {{
        let candidate = candidates[candidate_idx];
        let pos = candidate.x;
        let p = candidate.y;  // Pattern index to verify
        
        if pos < uniforms.input_len && p < uniforms.pattern_count {{
            let pat_offset = pattern_offsets[p].x;
            let pat_len = pattern_offsets[p].y;
            
            if pos + pat_len <= uniforms.input_len {{
                // Local position within the tile (in bytes)
                let local_pos = pos - tile_base;
                
                // If pattern extends beyond ACTUAL loaded tile bytes, fall back to global memory.
                // The tile may have fewer bytes at the end of the input — zeros fill the rest
                // which would cause false negatives if we read from the tile.
                let actual_tile_bytes = min(tile_words * 4u, uniforms.input_len - tile_base);
                let use_tile = (local_pos + pat_len) <= actual_tile_bytes;

                var matched = true;
                for (var i = 0u; i < pat_len; i = i + 1u) {{
                    var input_byte: u32;
                    if use_tile {{
                        let tile_byte_idx = local_pos + i;
                        let tile_word_idx = tile_byte_idx >> 2u;
                        let tile_shift = byte_shifts[tile_byte_idx & 3u];
                        input_byte = (input_tile[tile_word_idx] >> tile_shift) & 0xFFu;
                    }} else {{
                        let input_byte_idx = pos + i;
                        let input_word_idx = input_byte_idx >> 2u;
                        let input_shift = byte_shifts[input_byte_idx & 3u];
                        input_byte = (input_data[input_word_idx] >> input_shift) & 0xFFu;
                    }}

                    let pat_byte_idx = pat_offset + i;
                    let pat_word_idx = pat_byte_idx >> 2u;
                    let pat_shift = byte_shifts[pat_byte_idx & 3u];
                    let pattern_byte = (pattern_bytes[pat_word_idx] >> pat_shift) & 0xFFu;

                    if input_byte != pattern_byte {{
                        matched = false;
                        break;
                    }}
                }}

                if matched {{
                    let idx = atomicAdd(&match_count[0], 1u);
                    if idx < uniforms.max_matches {{
                        match_output[idx] = vec3<u32>(p, pos, pos + pat_len);
                    }} else {{
                        atomicStore(&match_count[1], 1u);
                    }}
                }}
            }}
        }}
        
        // Stride to next candidate for this thread
        candidate_idx = candidate_idx + total_threads;
    }}
}}"#
    )
}

/// Generate the regex DFA shader.
pub fn generate_regex_dfa_shader() -> String {
    format!(
        r#"struct Uniforms {{
    input_len: u32,
    start_state: u32,
    max_matches: u32,
    class_count: u32,
    eoi_class: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    byte_classes: array<vec4<u32>, 64>,
}}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read> transition_table: array<u32>;
@group(0) @binding(2) var<storage, read> match_list_pointers: array<u32>;
@group(0) @binding(3) var<storage, read> match_lists: array<u32>;
@group(0) @binding(4) var<storage, read_write> match_output: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> match_count: array<atomic<u32>, 2>;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;
@group(0) @binding(7) var<storage, read> pattern_lengths: array<u32>;

const FLAG_MATCH: u32 = 0x80000000u;
const FLAG_DEAD: u32 = 0x40000000u;
const MASK_STATE: u32 = 0x3FFFFFFFu;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pos = gid.x;
    if pos >= uniforms.input_len {{
        return;
    }}

    var state = uniforms.start_state;
    var found_match = false;
    // SECURITY: The regex DFA shader has O(N^2) worst-case complexity because each
    // thread starts at a different position and scans until the end of the input.
    // For large inputs (e.g. 128MB), this can cause GPU TDR.
    // FIX: Limit the maximum scan depth to 16MB to prevent unbounded execution.
    let end_limit = min(uniforms.input_len, pos + 16777216u);
    var last_match_ptr: u32 = 0xFFFFFFFFu;
    var last_match_i: u32 = 0u;

    for (var i = pos; i < end_limit; i = i + 1u) {{
        let word_idx = i >> 2u;
        let byte_offset = (i & 3u) * 8u;
        let byte = (input_data[word_idx] >> byte_offset) & 0xFFu;

        let class_id = uniforms.byte_classes[byte >> 2u][byte & 3u];
        let state_idx = state & MASK_STATE;
        state = transition_table[state_idx * uniforms.class_count + class_id];

        if (state & FLAG_MATCH) != 0u {{
            last_match_ptr = match_list_pointers[state & MASK_STATE];
            last_match_i = i;
        }}

        if (state & FLAG_DEAD) != 0u {{
            break;
        }}
    }}

    if (state & FLAG_DEAD) == 0u {{
        let state_idx = state & MASK_STATE;
        let eoi_state = transition_table[state_idx * uniforms.class_count + uniforms.eoi_class];
        if (eoi_state & FLAG_MATCH) != 0u {{
            last_match_ptr = match_list_pointers[eoi_state & MASK_STATE];
            last_match_i = end_limit - 1u;
        }}
    }}

    if last_match_ptr != 0xFFFFFFFFu {{
        let match_qty = match_lists[last_match_ptr];
        for (var m = 0u; m < match_qty; m = m + 1u) {{
            let count = atomicAdd(&match_count[0], 1u);
            if count < uniforms.max_matches {{
                let pat_id = match_lists[last_match_ptr + 1u + m];
                let pat_len = pattern_lengths[pat_id];
                let end_pos = select(last_match_i + 1u, min(pos + pat_len, uniforms.input_len), pat_len != 0u);
                match_output[count] = vec4<u32>(pat_id, pos, end_pos, 0u);
            }} else {{
                atomicStore(&match_count[1], 1u);
            }}
        }}
    }}
}}"#
    )
}

/// Maximum transitions that can be embedded as WGSL constants.
/// 16384 entries × 4 bytes = 64KB — well within shader limits.
/// Beyond this, fall back to the buffer-based `generate_regex_dfa_shader`.
const MAX_SPECIALIZED_TRANSITIONS: usize = 16_384;

/// Generate a specialized regex DFA shader with transitions embedded as WGSL constants.
///
/// Instead of reading from a storage buffer, the transition table, match list
/// pointers, match lists, pattern lengths, and byte classes are all emitted as
/// `const` arrays in the shader source. The GPU compiler can then:
/// - Eliminate dead states entirely (unreachable const entries pruned)
/// - Merge common transitions (constant propagation)
/// - Replace buffer loads with register-immediate moves
///
/// Returns `None` if the DFA is too large for constant embedding.
pub fn generate_specialized_dfa_shader(
    transition_table: &[u32],
    match_list_pointers: &[u32],
    match_lists: &[u32],
    pattern_lengths: &[u32],
    byte_classes: &[u32; 256],
    start_state: u32,
    class_count: u32,
    eoi_class: u32,
) -> Option<String> {
    if transition_table.len() > MAX_SPECIALIZED_TRANSITIONS {
        return None;
    }

    let transitions_str = format_const_array("TRANSITIONS", transition_table);
    let match_ptrs_str = format_const_array("MATCH_PTRS", match_list_pointers);
    let match_lists_str = format_const_array("MATCH_LISTS", match_lists);
    let pat_lengths_str = format_const_array("PAT_LENGTHS", pattern_lengths);

    // Byte classes as packed vec4<u32> array (64 entries of 4 values each)
    let mut packed_classes = [[0u32; 4]; 64];
    for (i, &c) in byte_classes.iter().enumerate() {
        packed_classes[i / 4][i % 4] = c;
    }
    let mut bc_entries = Vec::with_capacity(64);
    for quad in &packed_classes {
        bc_entries.push(format!(
            "vec4<u32>({}u, {}u, {}u, {}u)",
            quad[0], quad[1], quad[2], quad[3]
        ));
    }
    let bc_str = format!(
        "const BYTE_CLASSES: array<vec4<u32>, 64> = array<vec4<u32>, 64>(\n    {}\n);",
        bc_entries.join(",\n    ")
    );

    let state_count = if class_count > 0 {
        transition_table.len() / class_count as usize
    } else {
        0
    };

    Some(format!(
        r#"struct Uniforms {{
    input_len: u32,
    max_matches: u32,
    _padding0: u32,
    _padding1: u32,
}}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> match_output: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> match_count: array<atomic<u32>, 2>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const START_STATE: u32 = {start_state}u;
const CLASS_COUNT: u32 = {class_count}u;
const EOI_CLASS: u32 = {eoi_class}u;
const STATE_COUNT: u32 = {state_count}u;
const FLAG_MATCH: u32 = 0x80000000u;
const FLAG_DEAD: u32 = 0x40000000u;
const MASK_STATE: u32 = 0x3FFFFFFFu;

{transitions_str}
{match_ptrs_str}
{match_lists_str}
{pat_lengths_str}
{bc_str}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pos = gid.x;
    if pos >= uniforms.input_len {{
        return;
    }}

    var state = START_STATE;
    let end_limit = min(uniforms.input_len, pos + 16777216u);
    var last_match_ptr: u32 = 0xFFFFFFFFu;
    var last_match_i: u32 = 0u;

    for (var i = pos; i < end_limit; i = i + 1u) {{
        let word_idx = i >> 2u;
        let byte_offset = (i & 3u) * 8u;
        let byte = (input_data[word_idx] >> byte_offset) & 0xFFu;

        let class_id = BYTE_CLASSES[byte >> 2u][byte & 3u];
        let state_idx = state & MASK_STATE;
        state = TRANSITIONS[state_idx * CLASS_COUNT + class_id];

        if (state & FLAG_MATCH) != 0u {{
            last_match_ptr = MATCH_PTRS[state & MASK_STATE];
            last_match_i = i;
        }}

        if (state & FLAG_DEAD) != 0u {{
            break;
        }}
    }}

    if (state & FLAG_DEAD) == 0u {{
        let state_idx = state & MASK_STATE;
        let eoi_state = TRANSITIONS[state_idx * CLASS_COUNT + EOI_CLASS];
        if (eoi_state & FLAG_MATCH) != 0u {{
            last_match_ptr = MATCH_PTRS[eoi_state & MASK_STATE];
            last_match_i = end_limit - 1u;
        }}
    }}

    if last_match_ptr != 0xFFFFFFFFu {{
        let match_qty = MATCH_LISTS[last_match_ptr];
        for (var m = 0u; m < match_qty; m = m + 1u) {{
            let count = atomicAdd(&match_count[0], 1u);
            if count < uniforms.max_matches {{
                let pat_id = MATCH_LISTS[last_match_ptr + 1u + m];
                let pat_len = PAT_LENGTHS[pat_id];
                let end_pos = select(last_match_i + 1u, min(pos + pat_len, uniforms.input_len), pat_len != 0u);
                match_output[count] = vec4<u32>(pat_id, pos, end_pos, 0u);
            }} else {{
                atomicStore(&match_count[1], 1u);
            }}
        }}
    }}
}}"#
    ))
}

/// Format a u32 slice as a WGSL `const` array declaration.
fn format_const_array(name: &str, data: &[u32]) -> String {
    if data.is_empty() {
        return format!("const {name}: array<u32, 1> = array<u32, 1>(0u);");
    }
    let entries: Vec<String> = data.iter().map(|v| format!("{v}u")).collect();
    // Split into lines of 16 entries for readability
    let mut lines = Vec::new();
    for chunk in entries.chunks(16) {
        lines.push(format!("    {}", chunk.join(", ")));
    }
    format!(
        "const {name}: array<u32, {}> = array<u32, {}>({}  \n);",
        data.len(),
        data.len(),
        lines.join(",\n"),
    )
}

/// Check whether a DFA is eligible for specialized constant-embedded shaders.
pub fn can_specialize_dfa(transition_table_len: usize) -> bool {
    transition_table_len <= MAX_SPECIALIZED_TRANSITIONS
}

/// Helper method to dump all WGSL shader strings for debug and inspection.
pub fn dump_all_shaders() -> std::collections::HashMap<&'static str, String> {
    let mut shaders = std::collections::HashMap::new();
    shaders.insert("literal_prefilter", generate_literal_prefilter_shader());
    shaders.insert("literal_verify", generate_literal_verify_shader());
    shaders.insert("regex_dfa", generate_regex_dfa_shader());
    shaders.insert(
        "rolling_hash",
        crate::shader_hash::generate_rolling_hash_shader(),
    );
    shaders.insert(
        "smem_dfa",
        crate::shader_smem::generate_regex_dfa_smem_shader(),
    );
    shaders.insert(
        "algebraic_map",
        crate::algebraic::shader::generate_map_shader(),
    );
    shaders.insert(
        "algebraic_scan",
        crate::algebraic::shader::generate_scan_shader(),
    );
    shaders.insert(
        "algebraic_extract",
        crate::algebraic::shader::generate_extract_shader(),
    );
    shaders
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_prefilter_shader_has_candidate_buffer() {
        let source = generate_literal_prefilter_shader();
        assert!(source.contains("candidates"));
        assert!(source.contains("bucket_entries"));
        assert!(source.contains("prefix_meta"));
    }

    #[test]
    fn literal_verify_shader_has_overflow_detection() {
        let source = generate_literal_verify_shader();
        assert!(source.contains("atomicStore"));
        assert!(source.contains("match_count"));
    }

    #[test]
    fn regex_shader_walks_transition_table() {
        let source = generate_regex_dfa_shader();
        assert!(source.contains("transition_table["));
        assert!(source.contains("match_list_pointers"));
    }

    #[test]
    fn specialized_dfa_shader_embeds_constants() {
        let transitions = vec![0u32, 1, 2, 3, 4, 5, 6, 7]; // 2 states × 4 classes
        let match_ptrs = vec![0xFFFFFFFF, 0];
        let match_lists = vec![1, 0]; // 1 match: pattern 0
        let pat_lengths = vec![3];
        let mut byte_classes = [0u32; 256];
        byte_classes[b'a' as usize] = 1;
        byte_classes[b'b' as usize] = 2;
        byte_classes[b'c' as usize] = 3;

        let source = generate_specialized_dfa_shader(
            &transitions,
            &match_ptrs,
            &match_lists,
            &pat_lengths,
            &byte_classes,
            0,
            4,
            3,
        );

        assert!(source.is_some(), "small DFA should be specializable");
        let source = source.unwrap();

        // Transitions are const, not storage buffer
        assert!(
            source.contains("const TRANSITIONS:"),
            "transitions should be const"
        );
        assert!(
            source.contains("const MATCH_PTRS:"),
            "match pointers should be const"
        );
        assert!(
            source.contains("const MATCH_LISTS:"),
            "match lists should be const"
        );
        assert!(
            source.contains("const PAT_LENGTHS:"),
            "pattern lengths should be const"
        );
        assert!(
            source.contains("const BYTE_CLASSES:"),
            "byte classes should be const"
        );
        assert!(
            source.contains("const START_STATE:"),
            "start state should be const"
        );
        assert!(
            source.contains("const CLASS_COUNT:"),
            "class count should be const"
        );

        // Should NOT have storage buffer bindings for DFA data
        assert!(
            !source.contains("transition_table"),
            "no transition buffer binding"
        );
        assert!(
            !source.contains("match_list_pointers"),
            "no match_list_pointers buffer"
        );
    }

    #[test]
    fn specialized_dfa_rejects_large_tables() {
        let transitions = vec![0u32; 20_000]; // > MAX_SPECIALIZED_TRANSITIONS
        let result =
            generate_specialized_dfa_shader(&transitions, &[0], &[0], &[0], &[0; 256], 0, 1, 0);
        assert!(result.is_none(), "large DFA should not be specializable");
    }

    #[test]
    fn can_specialize_dfa_threshold() {
        assert!(can_specialize_dfa(100));
        assert!(can_specialize_dfa(16_384));
        assert!(!can_specialize_dfa(16_385));
    }

    #[test]
    fn format_const_array_handles_empty() {
        let result = format_const_array("EMPTY", &[]);
        assert!(result.contains("array<u32, 1>"));
        assert!(result.contains("0u"));
    }

    #[test]
    fn format_const_array_formats_values() {
        let result = format_const_array("TEST", &[42, 99, 0]);
        assert!(result.contains("array<u32, 3>"));
        assert!(result.contains("42u"));
        assert!(result.contains("99u"));
        assert!(result.contains("0u"));
    }
}
