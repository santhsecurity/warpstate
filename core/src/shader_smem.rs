//! WGSL compute shader generation for SMEM-staged DFA matching.
//!
//! # Shader Transparency
//!
//! - **Algorithm**: Dense Deterministic Finite Automaton (DFA) matching utilizing GPU Shared Memory (SMEM) for fast cache access.
//! - **Time Complexity**: O(N) for linear byte traversal over the matched sequence.
//! - **GPU Memory Requirements**: Bounded `MAX_SMEM_ENTRIES` for local cache, remaining dependencies handled by device memory mappings.
//! - **Compilation Strategy**: Runtime WGSL generation. Uses hardcoded template strings with safe internal property substitution (e.g., `WORKGROUP_SIZE`).

use crate::shader::WORKGROUP_SIZE;

/// Shared-memory table capacity in `u32` entries.
///
/// Most modern GPUs (Nvidia, AMD, Apple Silicon) provide at least 32KB of SMEM per workgroup,
/// often 48KB, 64KB, or more. We choose 8192 `u32` entries (32KB) as a robust,
/// conservative baseline that avoids maxing out SMEM limits and maintains high occupancy.
pub const MAX_SMEM_ENTRIES: u32 = 8192;
/// DFA state-count threshold below which transitions are baked directly into WGSL.
pub const MAX_SPECIALIZED_STATES: usize = 32;

/// Runtime status code indicating the shader completed normally.
pub const STATUS_OK: u32 = 0;

/// Runtime status code indicating the DFA table exceeded SMEM capacity.
pub const STATUS_TABLE_TOO_LARGE: u32 = 1;

const SHADER_PREAMBLE_TEMPLATE: &str = r#"struct Uniforms {
    input_len: u32,
    start_state: u32,
    max_matches: u32,
    class_count: u32,
    max_scan_depth: u32,
    eoi_class: u32,
    table_size: u32,
    _padding0: u32,
    byte_classes: array<vec4<u32>, 64>,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
__TRANSITION_BINDING__
@group(0) @binding(2) var<storage, read> match_list_pointers: array<u32>;
@group(0) @binding(3) var<storage, read> match_lists: array<u32>;
@group(0) @binding(4) var<storage, read_write> match_output: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> match_count: array<atomic<u32>, 3>;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;
@group(0) @binding(7) var<storage, read> pattern_lengths: array<u32>;

const FLAG_MATCH: u32 = 0x80000000u;
const FLAG_DEAD: u32 = 0x40000000u;
const MASK_STATE: u32 = 0x3FFFFFFFu;
const STATUS_OK: u32 = __STATUS_OK__u;
const STATUS_TABLE_TOO_LARGE: u32 = __STATUS_TABLE_TOO_LARGE__u;
const MAX_SMEM_ENTRIES: u32 = __MAX_SMEM_ENTRIES__u;
"#;

const REGEX_DFA_SMEM_SHADER_TEMPLATE: &str = r#"// warpstate — GPU Dense DFA Regex Shader with shared-memory staging

__PREAMBLE__

var<workgroup> smem_transitions: array<u32, MAX_SMEM_ENTRIES>;

fn load_table_into_smem(local_idx: u32) {
    let entries_per_thread = (uniforms.table_size + __WORKGROUP_SIZE__u - 1u) / __WORKGROUP_SIZE__u;
    for (var i = 0u; i < entries_per_thread; i = i + 1u) {
        let idx = local_idx + i * __WORKGROUP_SIZE__u;
        if idx < uniforms.table_size {
            smem_transitions[idx] = transition_table[idx];
        }
    }
}

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // CPU already initializes match_count[2] to STATUS_OK before dispatch.
    // Any thread in any workgroup can set error status (atomic).
    workgroupBarrier();
    if uniforms.table_size > MAX_SMEM_ENTRIES {
        atomicStore(&match_count[2], STATUS_TABLE_TOO_LARGE);
    }
    workgroupBarrier();

    if atomicLoad(&match_count[2]) != STATUS_OK {
        return;
    }

    load_table_into_smem(local_id.x);
    workgroupBarrier();

    let pos = gid.x + (gid.y * 65535u * __WORKGROUP_SIZE__u);
    if pos >= uniforms.input_len {
        return;
    }

    var state = uniforms.start_state;
    let end_limit = min(uniforms.input_len, pos + uniforms.max_scan_depth);

    for (var i = pos; i < end_limit; i = i + 1u) {
        let word_idx = i >> 2u;
        let byte_offset = (i & 3u) * 8u;
        let byte = (input_data[word_idx] >> byte_offset) & 0xFFu;

        let class_id = uniforms.byte_classes[byte >> 2u][byte & 3u];
        let state_idx = state & MASK_STATE;
        let smem_idx = state_idx * uniforms.class_count + class_id;
        if smem_idx >= MAX_SMEM_ENTRIES || smem_idx >= uniforms.table_size {
            atomicStore(&match_count[2], STATUS_TABLE_TOO_LARGE);
            return;
        }
        state = smem_transitions[smem_idx];

        if (state & FLAG_MATCH) != 0u {
            let new_state_idx = state & MASK_STATE;
            let match_ptr = match_list_pointers[new_state_idx];
            let match_qty = match_lists[match_ptr];

            for (var m = 0u; m < match_qty; m = m + 1u) {
                let count = atomicAdd(&match_count[0], 1u);
                if count < uniforms.max_matches {
                    let pat_id = match_lists[match_ptr + 1u + m];
                    let pat_len = pattern_lengths[pat_id];
                    let end_pos = select(i + 1u, min(pos + pat_len, uniforms.input_len), pat_len != 0u);
                    match_output[count] = vec4<u32>(pat_id, pos, end_pos, 0u);
                } else {
                    atomicStore(&match_count[1], 1u);
                }
            }
        }

        if (state & FLAG_DEAD) != 0u {
            break;
        }
    }

    if (state & FLAG_DEAD) == 0u {
        let state_idx = state & MASK_STATE;
        let eoi_smem_idx = state_idx * uniforms.class_count + uniforms.eoi_class;
        if eoi_smem_idx >= MAX_SMEM_ENTRIES || eoi_smem_idx >= uniforms.table_size {
            atomicStore(&match_count[2], STATUS_TABLE_TOO_LARGE);
            return;
        }
        state = smem_transitions[eoi_smem_idx];

        if (state & FLAG_MATCH) != 0u {
            let new_state_idx = state & MASK_STATE;
            let match_ptr = match_list_pointers[new_state_idx];
            let match_qty = match_lists[match_ptr];

            for (var m = 0u; m < match_qty; m = m + 1u) {
                let count = atomicAdd(&match_count[0], 1u);
                if count < uniforms.max_matches {
                    let pat_id = match_lists[match_ptr + 1u + m];
                    let pat_len = pattern_lengths[pat_id];
                    let end_pos = select(uniforms.input_len, min(pos + pat_len, uniforms.input_len), pat_len != 0u);
                    match_output[count] = vec4<u32>(pat_id, pos, end_pos, 0u);
                } else {
                    atomicStore(&match_count[1], 1u);
                }
            }
        }
    }
}"#;

const REGEX_DFA_SPECIALIZED_SHADER_TEMPLATE: &str = r#"// warpstate — GPU Dense DFA Regex Shader with baked transitions

__PREAMBLE__

const TRANS: array<u32, __TRANS_LEN__> = array<u32, __TRANS_LEN__>(__TRANS_DATA__);

fn transition_at(state_idx: u32, class_id: u32) -> u32 {
    let idx = state_idx * uniforms.class_count + class_id;
    if idx >= __TRANS_LEN__u {
        return FLAG_DEAD;
    }
    return TRANS[idx];
}

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let pos = gid.x + (gid.y * 65535u * __WORKGROUP_SIZE__u);
    if pos >= uniforms.input_len {
        return;
    }

    var state = uniforms.start_state;
    let end_limit = min(uniforms.input_len, pos + uniforms.max_scan_depth);

    for (var i = pos; i < end_limit; i = i + 1u) {
        let word_idx = i >> 2u;
        let byte_offset = (i & 3u) * 8u;
        let byte = (input_data[word_idx] >> byte_offset) & 0xFFu;

        let class_id = uniforms.byte_classes[byte >> 2u][byte & 3u];
        let state_idx = state & MASK_STATE;
        state = transition_at(state_idx, class_id);

        if (state & FLAG_MATCH) != 0u {
            let new_state_idx = state & MASK_STATE;
            let match_ptr = match_list_pointers[new_state_idx];
            let match_qty = match_lists[match_ptr];

            for (var m = 0u; m < match_qty; m = m + 1u) {
                let count = atomicAdd(&match_count[0], 1u);
                if count < uniforms.max_matches {
                    let pat_id = match_lists[match_ptr + 1u + m];
                    let pat_len = pattern_lengths[pat_id];
                    let end_pos = select(i + 1u, min(pos + pat_len, uniforms.input_len), pat_len != 0u);
                    match_output[count] = vec4<u32>(pat_id, pos, end_pos, 0u);
                } else {
                    atomicStore(&match_count[1], 1u);
                }
            }
        }

        if (state & FLAG_DEAD) != 0u {
            break;
        }
    }

    if (state & FLAG_DEAD) == 0u {
        let state_idx = state & MASK_STATE;
        state = transition_at(state_idx, uniforms.eoi_class);

        if (state & FLAG_MATCH) != 0u {
            let new_state_idx = state & MASK_STATE;
            let match_ptr = match_list_pointers[new_state_idx];
            let match_qty = match_lists[match_ptr];

            for (var m = 0u; m < match_qty; m = m + 1u) {
                let count = atomicAdd(&match_count[0], 1u);
                if count < uniforms.max_matches {
                    let pat_id = match_lists[match_ptr + 1u + m];
                    let pat_len = pattern_lengths[pat_id];
                    let end_pos = select(uniforms.input_len, min(pos + pat_len, uniforms.input_len), pat_len != 0u);
                    match_output[count] = vec4<u32>(pat_id, pos, end_pos, 0u);
                } else {
                    atomicStore(&match_count[1], 1u);
                }
            }
        }
    }
}"#;

fn shader_preamble(transition_binding: &str) -> String {
    SHADER_PREAMBLE_TEMPLATE
        .replace("__TRANSITION_BINDING__", transition_binding)
        .replace("__STATUS_OK__", &STATUS_OK.to_string())
        .replace(
            "__STATUS_TABLE_TOO_LARGE__",
            &STATUS_TABLE_TOO_LARGE.to_string(),
        )
        .replace("__MAX_SMEM_ENTRIES__", &MAX_SMEM_ENTRIES.to_string())
}

fn render_transition_data(transition_table: &[u32]) -> String {
    transition_table
        .iter()
        .map(|entry| format!("{entry}u"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate the regex DFA shader that stages the transition table in SMEM.
#[must_use]
pub fn generate_regex_dfa_smem_shader() -> String {
    REGEX_DFA_SMEM_SHADER_TEMPLATE
        .replace(
            "__PREAMBLE__",
            &shader_preamble(
                "@group(0) @binding(1) var<storage, read> transition_table: array<u32>;",
            ),
        )
        .replace("__WORKGROUP_SIZE__", &WORKGROUP_SIZE.to_string())
}

/// Return whether the DFA is small enough to benefit from transition specialization.
#[must_use]
pub const fn should_specialize_shader(state_count: usize) -> bool {
    state_count <= MAX_SPECIALIZED_STATES
}

/// Generate a regex DFA shader with transitions baked in as WGSL constants.
#[must_use]
pub fn generate_specialized_shader(transition_table: &[u32], class_count: u32) -> String {
    let transition_binding = "// Binding retained for layout compatibility across shader variants.\n@group(0) @binding(1) var<storage, read> transition_table_unused: array<u32>;";
    let transition_len = transition_table.len();
    let state_count = if class_count == 0 {
        0
    } else {
        transition_len / class_count as usize
    };
    let shader = REGEX_DFA_SPECIALIZED_SHADER_TEMPLATE
        .replace("__PREAMBLE__", &shader_preamble(transition_binding))
        .replace("__TRANS_LEN__", &transition_len.to_string())
        .replace("__TRANS_DATA__", &render_transition_data(transition_table))
        .replace("__WORKGROUP_SIZE__", &WORKGROUP_SIZE.to_string());
    format!("{shader}\n// specialized_state_count: {state_count}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smem_shader_declares_workgroup_table() {
        let source = generate_regex_dfa_smem_shader();
        assert!(source.contains("var<workgroup> smem_transitions"));
        assert!(source.contains("workgroupBarrier"));
    }

    #[test]
    fn smem_shader_exposes_runtime_status() {
        let source = generate_regex_dfa_smem_shader();
        assert!(source.contains("STATUS_TABLE_TOO_LARGE"));
        assert!(source.contains("match_count: array<atomic<u32>, 3>"));
    }

    #[test]
    fn specialization_threshold_is_state_based() {
        assert!(should_specialize_shader(MAX_SPECIALIZED_STATES));
        assert!(!should_specialize_shader(MAX_SPECIALIZED_STATES + 1));
    }

    #[test]
    fn specialized_shader_bakes_transition_constants() {
        let source = generate_specialized_shader(&[1, 2, 3, 4], 2);
        assert!(source.contains("const TRANS: array<u32, 4>"));
        assert!(source.contains("array<u32, 4>(1u, 2u, 3u, 4u)"));
        assert!(source.contains("transition_at(state_idx, class_id)"));
    }

    #[test]
    fn specialized_shader_skips_workgroup_staging() {
        let source = generate_specialized_shader(&[1, 2, 3, 4], 2);
        assert!(!source.contains("var<workgroup> smem_transitions"));
        assert!(!source.contains("load_table_into_smem"));
        assert!(source.contains("transition_table_unused"));
    }

    #[test]
    fn smem_shader_no_unconditional_status_ok() {
        let source = generate_regex_dfa_smem_shader();
        assert!(
            !source.contains("atomicStore(&match_count[2], STATUS_OK)"),
            "shader must not unconditionally overwrite status — CPU initializes it"
        );
    }

    #[test]
    fn smem_shader_checks_table_size() {
        let source = generate_regex_dfa_smem_shader();
        assert!(
            source.contains("smem_idx >= uniforms.table_size"),
            "SMEM index must be validated against actual table size, not just MAX_SMEM_ENTRIES"
        );
    }

    #[test]
    fn smem_shader_eoi_uses_input_len() {
        let source = generate_regex_dfa_smem_shader();
        assert!(
            source.contains("select(uniforms.input_len,"),
            "EOI match end position must use actual input_len, not end_limit"
        );
    }

    #[test]
    fn smem_specialized_transition_has_bounds_check() {
        let source = generate_specialized_shader(&[1, 2, 3, 4], 2);
        // __TRANS_LEN__ is replaced with the actual length (4), so the guard becomes "idx >= 4u".
        assert!(
            source.contains("idx >= 4u"),
            "specialized transition_at must guard against OOB"
        );
        assert!(source.contains("return FLAG_DEAD;"));
    }
}
