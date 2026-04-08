//! WGSL compute shader generation for rolling-hash pattern matching.
//!
//! # Shader Transparency
//!
//! - **Algorithm**: GPU-accelerated Rabin-Karp style rolling hash using FNV-1a.
//! - **Time Complexity**: O(N) average case for scanning.
//! - **GPU Memory Requirements**: Needs hash table buffer bounded by pattern count.
//! - **Compilation Strategy**: Runtime WGSL generation. Uses hardcoded template strings with safe `WORKGROUP_SIZE` substitution (no user-string injection vulnerability).

/// Workgroup size for rolling-hash compute shaders.
pub const WORKGROUP_SIZE: u32 = 256;

/// Generate the WGSL shader for the rolling-hash backend.
pub fn generate_rolling_hash_shader() -> String {
    format!(
        r#"struct Uniforms {{
    input_len: u32,
    pattern_length: u32,
    hash_table_size: u32,
    max_matches: u32,
}}

struct HashTableEntry {{
    occupied: u32,
    hash: u32,
    pattern_id: u32,
    padding: u32,
}}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read> hash_table: array<HashTableEntry>;
@group(0) @binding(2) var<storage, read> pattern_bytes: array<u32>;
@group(0) @binding(3) var<storage, read> pattern_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> match_output: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> match_count: array<atomic<u32>, 2>;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

const FNV_OFFSET_BASIS: u32 = 2166136261u;
const FNV_PRIME: u32 = 16777619u;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pos = gid.x + (gid.y * 65535u * {WORKGROUP_SIZE}u);
    if (pos + uniforms.pattern_length > uniforms.input_len) {{
        return;
    }}

    var hash = FNV_OFFSET_BASIS;
    for (var i = 0u; i < uniforms.pattern_length; i = i + 1u) {{
        let input_idx = pos + i;
        let input_word = input_data[input_idx >> 2u];
        let input_shift = (input_idx & 3u) * 8u;
        let byte_val = (input_word >> input_shift) & 0xFFu;
        hash = (hash ^ byte_val) * FNV_PRIME;
    }}

    let table_mask = uniforms.hash_table_size - 1u;
    for (var probe = 0u; probe < uniforms.hash_table_size; probe = probe + 1u) {{
        let slot = (hash + probe) & table_mask;
        let entry = hash_table[slot];
        if (entry.occupied == 0u) {{
            break;
        }}
        if (entry.hash != hash) {{
            continue;
        }}

        let pat_offset = pattern_offsets[entry.pattern_id];
        var verified = true;
        for (var j = 0u; j < uniforms.pattern_length; j = j + 1u) {{
            let input_idx = pos + j;
            let input_word = input_data[input_idx >> 2u];
            let input_shift = (input_idx & 3u) * 8u;
            let input_byte = (input_word >> input_shift) & 0xFFu;

            let pattern_idx = pat_offset + j;
            let pattern_word = pattern_bytes[pattern_idx >> 2u];
            let pattern_shift = (pattern_idx & 3u) * 8u;
            let pattern_byte = (pattern_word >> pattern_shift) & 0xFFu;

            if (input_byte != pattern_byte) {{
                verified = false;
                break;
            }}
        }}

        if (verified) {{
            let idx = atomicAdd(&match_count[0], 1u);
            if (idx < uniforms.max_matches) {{
                match_output[idx] = vec4<u32>(entry.pattern_id, pos, pos + uniforms.pattern_length, 0u);
            }} else {{
                atomicStore(&match_count[1], 1u);
            }}
        }}
    }}
}}"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_hash_shader_has_hash_table_entry() {
        let source = generate_rolling_hash_shader();
        assert!(source.contains("struct HashTableEntry"));
        assert!(source.contains("occupied"));
    }

    #[test]
    fn rolling_hash_shader_verifies_bytes() {
        let source = generate_rolling_hash_shader();
        assert!(source.contains("pattern_offsets"));
        assert!(source.contains("verified = false"));
    }

    #[test]
    fn rolling_hash_shader_reports_overflow() {
        let source = generate_rolling_hash_shader();
        assert!(source.contains("atomicAdd"));
        assert!(source.contains("atomicStore(&match_count[1], 1u)"));
    }
}
