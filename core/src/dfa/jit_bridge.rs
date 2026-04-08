use super::{RegexDFA, MASK_STATE};
use dfajit::{JitDfa, TransitionTable};
use std::sync::Arc;

pub(crate) fn compile_jit_dfa(
    transition_table: &[u32],
    match_list_pointers: &[u32],
    match_lists: &[u32],
    pattern_lengths: &[u32],
    pattern_strings: Option<&[&str]>,
    start_state: u32,
    class_count: u32,
    byte_classes: &[u32; 256],
) -> Option<Arc<JitDfa>> {
    if transition_table.is_empty() || class_count == 0 {
        return None;
    }
    if let Some(patterns) = pattern_strings {
        if !patterns_are_jit_safe(patterns) {
            return None;
        }
    }
    let state_count = transition_table.len() / class_count as usize;
    if state_count == 0 {
        return None;
    }

    let start_idx = start_state & MASK_STATE;
    let start_idx_usize = start_idx as usize;
    let mut original_for_jit = Vec::with_capacity(state_count);
    original_for_jit.push(start_idx_usize);
    for state_idx in 0..state_count {
        if state_idx != start_idx_usize {
            original_for_jit.push(state_idx);
        }
    }
    let mut jit_for_original = vec![0usize; state_count];
    for (jit_idx, &original_idx) in original_for_jit.iter().enumerate() {
        jit_for_original[original_idx] = jit_idx;
    }

    let mut table = TransitionTable::new(state_count, 256).ok()?;
    for (jit_idx, &original_idx) in original_for_jit.iter().enumerate() {
        for byte in 0..=u8::MAX {
            let class_id = byte_classes[usize::from(byte)] as usize;
            let mut next = transition_for_class_raw(
                transition_table,
                class_count,
                original_idx as u32,
                class_id,
            );
            if RegexDFA::is_dead_state(next) {
                let restart =
                    transition_for_class_raw(transition_table, class_count, start_idx, class_id);
                next = if RegexDFA::is_dead_state(restart) {
                    start_idx
                } else {
                    restart
                };
            }
            table.set_transition(
                jit_idx,
                byte,
                jit_for_original[(next & MASK_STATE) as usize] as u32,
            );
        }
    }

    for (original_idx, &ptr) in match_list_pointers.iter().enumerate() {
        let ptr = ptr as usize;
        let &qty = match_lists.get(ptr)?;
        for offset in 0..qty as usize {
            let &pattern_id = match_lists.get(ptr + 1 + offset)?;
            let &pattern_len = pattern_lengths.get(pattern_id as usize)?;
            if pattern_len == 0 {
                return None;
            }
            table.add_accept(jit_for_original[original_idx] as u32, pattern_id);
        }
    }

    for (pattern_id, &length) in pattern_lengths.iter().enumerate() {
        if length > 0 {
            table.set_pattern_length(pattern_id as u32, length);
        }
    }

    JitDfa::compile(&table).ok().map(Arc::new)
}

fn transition_for_class_raw(
    transition_table: &[u32],
    class_count: u32,
    state: u32,
    class_id: usize,
) -> u32 {
    let state_idx = (state & MASK_STATE) as usize;
    transition_table[state_idx * class_count as usize + class_id]
}

fn patterns_are_jit_safe(patterns: &[&str]) -> bool {
    patterns.iter().all(|pattern| {
        !pattern.contains('^')
            && !pattern.contains('$')
            && !pattern.contains("\\A")
            && !pattern.contains("\\z")
            && !pattern.contains("\\Z")
    })
}
