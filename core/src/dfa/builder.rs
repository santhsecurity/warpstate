use std::collections::{HashMap, VecDeque};

use regex_automata::dfa::{dense, Automaton};
use regex_automata::util::primitives::StateID;
use regex_automata::{Anchored, Input};
use regex_syntax::hir::{Hir, HirKind};

use super::{alloc_huge_page_vec_with_backing, RegexDFA, FLAG_DEAD, FLAG_MATCH, MAX_DFA_STATES};
use crate::error::{Error, Result};

impl RegexDFA {
    /// Build a dense Regex DFA.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn build(pattern_strings: &[&str], original_ids: &[usize]) -> Result<Self> {
        // Build with anchored-only start kind. The GPU shader starts a separate
        // DFA walk at every byte position, so the start state MUST reject bytes
        // that don't match the pattern's first character (no implicit `.*` prefix).
        let compiled = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .dfa_size_limit(Some(50_000_000))
                    .match_kind(regex_automata::MatchKind::All)
                    // Both: GPU needs anchored (walks from every byte),
                    // CPU native scan needs anchored per-position search.
                    .start_kind(regex_automata::dfa::StartKind::Both),
            )
            .syntax(
                regex_automata::util::syntax::Config::new()
                    .multi_line(true)
                    .unicode(false)
                    .utf8(false),
            )
            .build_many(pattern_strings)
            .map_err(|error| Error::PatternCompilationFailed {
                reason: format!("regex compilation failed: {error}"),
            })?;

        let classes = compiled.byte_classes();
        // alphabet_len is bounded by regex-automata internals (typically <= 256)
        #[allow(clippy::cast_possible_truncation)]
        let class_count = classes.alphabet_len() as u32;
        // eoi class is bounded by alphabet_len which is <= 256
        #[allow(clippy::cast_possible_truncation)]
        let eoi_class = classes.eoi().as_usize() as u32;

        let mut byte_classes = [0u32; 256];
        let mut sample_bytes = vec![0u8; class_count as usize];
        let mut compact_byte_classes = [0u8; 256];

        for b in 0..=u8::MAX {
            let c = classes.get(b);
            byte_classes[b as usize] = c as u32;
            compact_byte_classes[b as usize] = c;
            if c as u32 != eoi_class {
                sample_bytes[c as usize] = b;
            }
        }

        let anchored_input = Input::new(b"").anchored(Anchored::Yes);
        let start_state = compiled
            .start_state_forward(&anchored_input)
            .map_err(|error| Error::PatternCompilationFailed {
                reason: format!("failed to resolve start state: {error}"),
            })?;

        let unanchored_input = Input::new(b"").anchored(Anchored::No);
        let start_unanchored =
            compiled
                .start_state_forward(&unanchored_input)
                .map_err(|error| Error::PatternCompilationFailed {
                    reason: format!("failed to resolve unanchored start state: {error}"),
                })?;

        let mut compact_states = Vec::new();
        let mut compact_state_to_idx = HashMap::<StateID, u32>::new();
        let mut compact_queue = VecDeque::new();

        push_state(
            start_unanchored,
            &mut compact_states,
            &mut compact_state_to_idx,
            &mut compact_queue,
        )?;
        if !compact_state_to_idx.contains_key(&start_state) {
            push_state(
                start_state,
                &mut compact_states,
                &mut compact_state_to_idx,
                &mut compact_queue,
            )?;
        }

        while let Some(state) = compact_queue.pop_front() {
            if compact_states.len() >= MAX_DFA_STATES {
                break; // Abandon compact optimization if states reach max
            }
            for c in 0..class_count {
                let next = if c == eoi_class {
                    compiled.next_eoi_state(state)
                } else {
                    compiled.next_state(state, sample_bytes[c as usize])
                };
                if !compact_state_to_idx.contains_key(&next) {
                    push_state(
                        next,
                        &mut compact_states,
                        &mut compact_state_to_idx,
                        &mut compact_queue,
                    )?;
                }
            }
        }

        let mut compact_dfa = None;
        if compact_states.len() <= MAX_DFA_STATES {
            let mut flags = vec![0u8; compact_states.len()];
            let mut match_pattern = vec![0u32; compact_states.len()];
            let mut trans_u32 = vec![0u32; compact_states.len() * class_count as usize];
            let mut trans_u8 = if compact_states.len() < 256 {
                Some(vec![0u8; compact_states.len() * class_count as usize])
            } else {
                None
            };
            let mut trans_u4 = if compact_states.len() < 16 {
                Some(vec![
                    0u8;
                    (compact_states.len() * class_count as usize)
                        .div_ceil(2)
                ])
            } else {
                None
            };

            for (i, &state) in compact_states.iter().enumerate() {
                let mut f = 0;
                if compiled.is_match_state(state) {
                    f |= 1;
                    let pat_idx = compiled.match_pattern(state, 0).as_usize();
                    match_pattern[i] = original_ids.get(pat_idx).copied().unwrap_or(pat_idx) as u32;
                }
                if compiled.is_dead_state(state) {
                    f |= 2;
                }
                if compiled.is_quit_state(state) {
                    f |= 4;
                }
                flags[i] = f;

                for c in 0..class_count {
                    let next = if c == eoi_class {
                        compiled.next_eoi_state(state)
                    } else {
                        compiled.next_state(state, sample_bytes[c as usize])
                    };
                    let next_idx = match compact_state_to_idx.get(&next) {
                        Some(&idx) => idx,
                        None => 0, // Dead state — treat as state 0 (start/reject)
                    };
                    let trans_idx = i * class_count as usize + c as usize;
                    trans_u32[trans_idx] = next_idx;
                    if let Some(t8) = &mut trans_u8 {
                        t8[trans_idx] = next_idx as u8;
                    }
                    if let Some(t4) = &mut trans_u4 {
                        let next_idx_u8 = next_idx as u8;
                        if trans_idx % 2 == 0 {
                            t4[trans_idx / 2] = (t4[trans_idx / 2] & 0xF0) | (next_idx_u8 & 0x0F);
                        } else {
                            t4[trans_idx / 2] =
                                (t4[trans_idx / 2] & 0x0F) | ((next_idx_u8 & 0x0F) << 4);
                        }
                    }
                }
            }

            compact_dfa = Some(crate::dfa::CompactDfa {
                trans_u4,
                trans_u8,
                trans_u32,
                flags,
                match_pattern,
                start_unanchored: compact_state_to_idx[&start_unanchored],
                start_anchored: compact_state_to_idx[&start_state],
                class_count: class_count as usize,
                eoi_class: eoi_class as usize,
                byte_classes: compact_byte_classes,
            });
        }

        let mut states = Vec::new();
        let mut state_to_index = HashMap::<StateID, u32>::new();
        let mut queue = VecDeque::new();
        push_state(start_state, &mut states, &mut state_to_index, &mut queue)?;

        while let Some(state) = queue.pop_front() {
            if states.len() > MAX_DFA_STATES {
                return Err(Error::PatternCompilationFailed {
                    reason: format!("regex Dfa exceeded {MAX_DFA_STATES} states"),
                });
            }
            for c in 0..class_count {
                let next = if c == eoi_class {
                    compiled.next_eoi_state(state)
                } else {
                    compiled.next_state(state, sample_bytes[c as usize])
                };
                if !state_to_index.contains_key(&next) {
                    push_state(next, &mut states, &mut state_to_index, &mut queue)?;
                }
            }
        }

        let mut match_lists = vec![0u32]; // 0 is empty list
        let mut match_list_pointers = vec![0u32; states.len()];
        for (index, &state) in states.iter().enumerate() {
            if compiled.is_match_state(state) {
                let m_len = compiled.match_len(state);
                // SAFETY: match_lists.len() is bounded by MAX_DFA_STATES which fits in u32
                match_list_pointers[index] = match_lists.len() as u32;
                // SAFETY: m_len is bounded by regex-automata internals
                match_lists.push(m_len as u32);
                for i in 0..m_len {
                    let pat_idx = compiled.match_pattern(state, i).as_usize();
                    // SAFETY: pattern IDs are bounded by original_ids length
                    match_lists.push(original_ids[pat_idx] as u32);
                }
            }
        }

        let transition_table_len = states.len() * class_count as usize;
        let mut transition_table = alloc_huge_page_vec_with_backing(transition_table_len);
        for &state in &states {
            for c in 0..class_count {
                let next = if c == eoi_class {
                    compiled.next_eoi_state(state)
                } else {
                    compiled.next_state(state, sample_bytes[c as usize])
                };
                let next_idx = state_to_index[&next];
                let mut flags = 0;
                if compiled.is_match_state(next) {
                    flags |= FLAG_MATCH;
                }
                if compiled.is_dead_state(next) {
                    flags |= FLAG_DEAD;
                }
                transition_table.0.push(next_idx | flags);
            }
        }

        let start_idx = state_to_index[&start_state];
        let mut start_flags = 0;
        if compiled.is_match_state(start_state) {
            start_flags |= FLAG_MATCH;
        }
        if compiled.is_dead_state(start_state) {
            start_flags |= FLAG_DEAD;
        }

        let max_id: usize = original_ids.iter().max().copied().unwrap_or_default();
        // Cap allocation to prevent OOM from corrupted/malicious pattern IDs.
        // 1M patterns is generous; real-world usage is typically < 100K.
        if max_id > 1_000_000 {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "pattern ID {max_id} exceeds maximum supported (1,000,000). Fix: use contiguous pattern IDs."
                ),
            });
        }
        let mut pattern_lengths = vec![0u32; max_id + 1];
        for (&pattern, &original_id) in pattern_strings.iter().zip(original_ids.iter()) {
            if let Some(length) = fixed_regex_length(pattern) {
                pattern_lengths[original_id] = u32::try_from(length).unwrap_or(0);
            }
        }
        let (native_dfa_bytes, _) = compiled.to_bytes_little_endian();
        #[cfg(feature = "jit")]
        let jit_dfa = super::jit_bridge::compile_jit_dfa(
            &transition_table.0,
            &match_list_pointers,
            &match_lists,
            &pattern_lengths,
            Some(pattern_strings),
            start_idx | start_flags,
            class_count,
            &byte_classes,
        );
        // Note: Regex lengths are variable. Setting pattern length to 0 is handled safely by GPU / CPU routines.
        Ok(Self {
            transition_table: transition_table.0,
            match_list_pointers,
            match_lists,
            pattern_lengths,
            start_state: start_idx | start_flags,
            class_count,
            eoi_class,
            byte_classes,
            native_dfa: Some(compiled),
            native_dfa_bytes,
            native_original_ids: original_ids.to_vec(),
            transition_table_backing: transition_table.1,
            // Build fast regex for CPU scanning — handles start+end in one pass
            // Only build fast_regex for single-pattern DFAs. Multi-pattern
            // combined alternation loses pattern identity and changes priority.
            fast_regex: if pattern_strings.len() == 1 {
                regex::bytes::RegexBuilder::new(pattern_strings[0])
                    .multi_line(true)
                    .build()
                    .ok()
            } else {
                None
            },
            // For multi-pattern: build per-pattern regex crate engines for k-way merge.
            // Each uses SIMD + lazy DFA independently, avoiding the O(256) backwards
            // start search that makes scan_native_with_dfa slow.
            fast_regexes: if pattern_strings.len() > 1 {
                pattern_strings
                    .iter()
                    .zip(original_ids.iter())
                    .filter_map(|(pat, &orig_id)| {
                        regex::bytes::RegexBuilder::new(pat)
                            .multi_line(true)
                            .build()
                            .ok()
                            // SAFETY: orig_id is bounded by original_ids length
                            .map(|re| (re, orig_id as u32))
                    })
                    .collect()
            } else {
                Vec::new()
            },
            compact_dfa,
            #[cfg(feature = "jit")]
            jit_dfa,
        })
    }
}

fn fixed_regex_length(pattern: &str) -> Option<usize> {
    let hir = regex_syntax::Parser::new().parse(pattern).ok()?;
    fixed_hir_length(&hir)
}

fn fixed_hir_length(hir: &Hir) -> Option<usize> {
    match hir.kind() {
        HirKind::Empty | HirKind::Look(_) => Some(0),
        HirKind::Literal(literal) => Some(literal.0.len()),
        HirKind::Concat(parts) => {
            let mut total = 0usize;
            for part in parts {
                total += fixed_hir_length(part)?;
            }
            Some(total)
        }
        HirKind::Alternation(parts) => {
            let mut lengths = parts.iter().filter_map(fixed_hir_length);
            let first = lengths.next()?;
            if lengths.all(|length| length == first) {
                Some(first)
            } else {
                None
            }
        }
        HirKind::Class(class) => match class {
            regex_syntax::hir::Class::Unicode(_) | regex_syntax::hir::Class::Bytes(_) => Some(1),
        },
        HirKind::Repetition(rep) => {
            let sub_len = fixed_hir_length(&rep.sub)?;
            match (rep.min, rep.max) {
                (min, Some(max)) if min == max => Some(sub_len * min as usize),
                _ => None,
            }
        }
        HirKind::Capture(capture) => fixed_hir_length(&capture.sub),
    }
}

fn push_state(
    state: StateID,
    states: &mut Vec<StateID>,
    state_to_index: &mut HashMap<StateID, u32>,
    queue: &mut VecDeque<StateID>,
) -> Result<()> {
    let index = states
        .len()
        .try_into()
        .map_err(|_| Error::PatternCompilationFailed {
            reason: "regex DFA exceeded state index range".to_string(),
        })?;
    states.push(state);
    state_to_index.insert(state, index);
    queue.push_back(state);
    Ok(())
}
