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

    let pos = gid.x + (gid.y * 65535u * {WORKGROUP_SIZE}u);
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
    let wg_start = (wg_id.x + (wg_id.y * 65535u)) * {WORKGROUP_SIZE}u;
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
    let total_threads = num_wg.x * num_wg.y * {WORKGROUP_SIZE}u;
    
    // Each thread processes candidates starting at gid.x, striding by total_threads.
    // This distributes the O(candidates) work across all threads in the dispatch.
    var candidate_idx = gid.x + (gid.y * 65535u * {WORKGROUP_SIZE}u);
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

/// Helper method to dump all WGSL shader strings for debug and inspection.
pub fn dump_all_shaders() -> std::collections::HashMap<&'static str, String> {
    let mut shaders = std::collections::HashMap::new();
    shaders.insert("literal_prefilter", generate_literal_prefilter_shader());
    shaders.insert("literal_verify", generate_literal_verify_shader());
    shaders.insert(
        "regex_ensemble",
        include_str!("ensemble_shader.wgsl").to_string(),
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
    fn dump_all_shaders_lists_consolidated_gpu_shaders() {
        let shaders = dump_all_shaders();
        assert!(shaders.contains_key("literal_prefilter"));
        assert!(shaders.contains_key("literal_verify"));
        assert!(shaders.contains_key("regex_ensemble"));
    }
}
