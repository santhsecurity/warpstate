struct DfaMetadata {
    offset_words: u32,
    state_count: u32,
    class_count: u32,
    start_state: u32,
}

struct Uniforms {
    input_length: u32,
    dfa_count: u32,
    _padding0: u32,
    _padding1: u32,
}

const WORKGROUP_SIZE: u32 = 256u;
const SHARED_TABLE_WORDS: u32 = 12288u;
const BYTE_CLASS_WORDS: u32 = 64u;
const FLAG_MATCH: u32 = 1u;
const FLAG_DEAD: u32 = 2u;

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read> dfa_tables: array<u32>;
@group(0) @binding(2) var<storage, read> dfa_metadata: array<DfaMetadata>;
@group(0) @binding(3) var<storage, read_write> match_output: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

var<workgroup> shared_table: array<u32, 12288>;
var<workgroup> shared_match: atomic<u32>;

fn read_input_byte(index: u32) -> u32 {
    let word = input_data[index >> 2u];
    let shift = (index & 3u) << 3u;
    return (word >> shift) & 0xFFu;
}

fn table_word(offset: u32, local_index: u32, use_shared: bool) -> u32 {
    if use_shared {
        return shared_table[local_index];
    }
    return dfa_tables[offset + local_index];
}

fn byte_class(offset: u32, state_count: u32, class_count: u32, byte_value: u32, use_shared: bool) -> u32 {
    let byte_words_offset = state_count * class_count + state_count;
    let packed_word = table_word(offset, byte_words_offset + (byte_value >> 2u), use_shared);
    let shift = (byte_value & 3u) << 3u;
    return (packed_word >> shift) & 0xFFu;
}

fn is_match_state(offset: u32, state_count: u32, class_count: u32, state: u32, use_shared: bool) -> bool {
    let flags_offset = state_count * class_count;
    return (table_word(offset, flags_offset + state, use_shared) & FLAG_MATCH) != 0u;
}

fn is_dead_state(offset: u32, state_count: u32, class_count: u32, state: u32, use_shared: bool) -> bool {
    let flags_offset = state_count * class_count;
    return (table_word(offset, flags_offset + state, use_shared) & FLAG_DEAD) != 0u;
}

fn next_state(offset: u32, class_count: u32, state: u32, class_id: u32, use_shared: bool) -> u32 {
    return table_word(offset, state * class_count + class_id, use_shared);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let dfa_index = workgroup_id.x;
    if dfa_index >= uniforms.dfa_count {
        return;
    }

    let dfa_meta = dfa_metadata[dfa_index];
    let blob_words = dfa_meta.state_count * dfa_meta.class_count + dfa_meta.state_count + BYTE_CLASS_WORDS;
    let use_shared = blob_words <= SHARED_TABLE_WORDS;

    if local_index == 0u {
        atomicStore(&shared_match, 0u);
    }

    if use_shared {
        var word_index = local_index;
        while word_index < blob_words {
            shared_table[word_index] = dfa_tables[dfa_meta.offset_words + word_index];
            word_index = word_index + WORKGROUP_SIZE;
        }
    }
    workgroupBarrier();

    var position = local_index;
    while position < uniforms.input_length {
        if atomicLoad(&shared_match) != 0u {
            break;
        }

        var state = dfa_meta.start_state;
        var cursor = position;

        loop {
            let class_id = byte_class(
                dfa_meta.offset_words,
                dfa_meta.state_count,
                dfa_meta.class_count,
                read_input_byte(cursor),
                use_shared,
            );
            state = next_state(dfa_meta.offset_words, dfa_meta.class_count, state, class_id, use_shared);
            if is_match_state(dfa_meta.offset_words, dfa_meta.state_count, dfa_meta.class_count, state, use_shared) {
                atomicStore(&shared_match, 1u);
                break;
            }
            if is_dead_state(dfa_meta.offset_words, dfa_meta.state_count, dfa_meta.class_count, state, use_shared) {
                break;
            }
            cursor = cursor + 1u;
            if cursor >= uniforms.input_length || state >= dfa_meta.state_count {
                break;
            }
        }

        position = position + WORKGROUP_SIZE;
    }

    workgroupBarrier();
    if local_index == 0u {
        match_output[dfa_index] = atomicLoad(&shared_match);
    }
}
