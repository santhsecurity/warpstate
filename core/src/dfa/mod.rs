//! Regex DFA compilation and execution.

#[cfg(feature = "jit")]
use std::sync::Arc;

#[cfg(feature = "jit")]
use dfajit::JitDfa;
use regex_automata::dfa::dense;

use crate::error::{Error, Result};

pub(crate) mod builder;
#[cfg(feature = "jit")]
pub(crate) mod jit_bridge;
pub(crate) mod scan;

pub(crate) const MAX_DFA_STATES: usize = 100_000;
pub(crate) const MAX_SCAN_MATCHES: usize = 1_048_576;
pub(crate) const FLAG_MATCH: u32 = 0x8000_0000;
pub(crate) const FLAG_DEAD: u32 = 0x4000_0000;
/// Mask to extract state index from a combined state/flag integer.
pub const MASK_STATE: u32 = 0x3FFF_FFFF;

/// Compact DFA for optimized CPU scanning.
///
/// Cache-line aligned (64 bytes) for optimal access patterns during scan.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct CompactDfa {
    /// 4-bit transition table (if states < 16).
    pub(crate) trans_u4: Option<Vec<u8>>,
    /// 8-bit transition table (if states < 256).
    pub(crate) trans_u8: Option<Vec<u8>>,
    /// 32-bit transition table fallback.
    pub(crate) trans_u32: Vec<u32>,
    /// State flags: 1 = match, 2 = dead, 4 = quit.
    pub(crate) flags: Vec<u8>,
    /// Match pattern ID for each state.
    pub(crate) match_pattern: Vec<u32>,
    /// Unanchored start state.
    pub(crate) start_unanchored: u32,
    /// Anchored start state.
    pub(crate) start_anchored: u32,
    /// Number of alphabet classes.
    pub(crate) class_count: usize,
    /// End of input class.
    pub(crate) eoi_class: usize,
    /// Byte to class mapping.
    pub(crate) byte_classes: [u8; 256],
}

impl CompactDfa {
    #[inline(always)]
    fn flag_byte(&self, state: u32) -> u8 {
        self.flags.get(state as usize).copied().unwrap_or(2)
    }

    #[inline(always)]
    #[must_use]
    pub fn is_match_state(&self, state: u32) -> bool {
        self.flag_byte(state) & 1 != 0
    }

    #[inline(always)]
    #[must_use]
    pub fn is_dead_state(&self, state: u32) -> bool {
        self.flag_byte(state) & 2 != 0
    }

    #[inline(always)]
    #[must_use]
    pub fn is_quit_state(&self, state: u32) -> bool {
        self.flag_byte(state) & 4 != 0
    }

    #[inline(always)]
    fn transition_index(&self, state: u32, class: usize) -> usize {
        let idx = (state as usize) * self.class_count + class;
        debug_assert!(idx / self.class_count == state as usize);
        idx
    }

    #[inline(always)]
    fn transition_u4(trans: &[u8], idx: usize) -> u32 {
        let packed_idx = idx / 2;
        let val = trans.get(packed_idx).copied().unwrap_or(0);
        if idx % 2 == 0 {
            (val & 0x0F) as u32
        } else {
            (val >> 4) as u32
        }
    }

    #[inline(always)]
    fn transition_u8(trans: &[u8], idx: usize) -> u32 {
        trans.get(idx).copied().unwrap_or(0) as u32
    }

    #[inline(always)]
    fn transition_u32(trans: &[u32], idx: usize) -> u32 {
        trans.get(idx).copied().unwrap_or(0)
    }

    #[inline(always)]
    pub fn next_state(&self, state: u32, byte: u8) -> u32 {
        let class = self.byte_classes[usize::from(byte)] as usize;
        let idx = self.transition_index(state, class);

        if let Some(trans) = &self.trans_u4 {
            Self::transition_u4(trans, idx)
        } else if let Some(trans) = &self.trans_u8 {
            Self::transition_u8(trans, idx)
        } else {
            Self::transition_u32(&self.trans_u32, idx)
        }
    }

    #[inline(always)]
    pub fn next_eoi_state(&self, state: u32) -> u32 {
        let idx = self.transition_index(state, self.eoi_class);
        if let Some(trans) = &self.trans_u4 {
            Self::transition_u4(trans, idx)
        } else if let Some(trans) = &self.trans_u8 {
            Self::transition_u8(trans, idx)
        } else {
            Self::transition_u32(&self.trans_u32, idx)
        }
    }
}

/// Dense DFA compiled from one or more patterns.
///
/// Cache-line aligned (64 bytes) for optimal access patterns during scan.
#[repr(align(64))]
#[derive(Debug)]
pub struct RegexDFA {
    /// Contiguous transition table (for GPU backend).
    pub transition_table: Vec<u32>,
    /// Offsets to match lists for each state.
    pub match_list_pointers: Vec<u32>,
    /// Contiguous match lists.
    pub match_lists: Vec<u32>,
    /// Pattern execution lengths.
    pub pattern_lengths: Vec<u32>,
    /// Initial DFA state.
    pub start_state: u32,
    /// Number of alphabet classes logic requires.
    pub class_count: u32,
    /// Class assigned to end-of-input logic.
    pub eoi_class: u32,
    /// Byte values mapping to character classes.
    pub byte_classes: [u32; 256],
    /// The native regex-automata DFA for correct CPU scanning.
    pub(crate) native_dfa: Option<dense::DFA<Vec<u32>>>,
    pub(crate) native_dfa_bytes: Vec<u8>,
    /// Mapping from native pattern indices to our original IDs.
    pub(crate) native_original_ids: Vec<usize>,
    pub(crate) transition_table_backing: TransitionTableBacking,
    /// Fast regex for CPU scanning — uses regex crate's optimized engine
    /// instead of byte-by-byte DFA walk + backwards start search.
    pub(crate) fast_regex: Option<regex::bytes::Regex>,
    /// Per-pattern fast regexes for multi-pattern k-way merge.
    /// Each entry is (compiled_regex, original_pattern_id).
    /// Uses regex crate's SIMD + lazy DFA per pattern, then merges by position.
    pub(crate) fast_regexes: Vec<(regex::bytes::Regex, u32)>,
    /// Compact DFA for high-performance CPU matching.
    pub(crate) compact_dfa: Option<CompactDfa>,
    /// Cached JIT-compiled DFA for fixed-length, non-EOI-safe scans.
    #[cfg(feature = "jit")]
    pub(crate) jit_dfa: Option<Arc<JitDfa>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransitionTableBacking {
    Standard,
    #[cfg(target_os = "linux")]
    HugePages {
        byte_len: usize,
    },
}

impl Clone for RegexDFA {
    fn clone(&self) -> Self {
        Self {
            transition_table: self.transition_table.clone(),
            match_list_pointers: self.match_list_pointers.clone(),
            match_lists: self.match_lists.clone(),
            pattern_lengths: self.pattern_lengths.clone(),
            start_state: self.start_state,
            class_count: self.class_count,
            eoi_class: self.eoi_class,
            byte_classes: self.byte_classes,
            native_dfa: self.native_dfa.clone(),
            native_dfa_bytes: self.native_dfa_bytes.clone(),
            native_original_ids: self.native_original_ids.clone(),
            transition_table_backing: self.transition_table_backing,
            fast_regex: self.fast_regex.clone(),
            fast_regexes: self.fast_regexes.clone(),
            compact_dfa: self.compact_dfa.clone(),
            #[cfg(feature = "jit")]
            jit_dfa: self.jit_dfa.clone(),
        }
    }
}

impl RegexDFA {
    #[inline(always)]
    fn byte_class(&self, byte: u8) -> usize {
        self.byte_classes[usize::from(byte)] as usize
    }

    #[inline(always)]
    fn transition_table_entry(&self, idx: usize) -> u32 {
        self.transition_table.get(idx).copied().unwrap_or(FLAG_DEAD)
    }

    #[inline]
    pub(crate) fn state_count(&self) -> usize {
        self.transition_table.len() / self.class_count as usize
    }

    #[inline(always)]
    pub(crate) fn is_match_state(state: u32) -> bool {
        (state & FLAG_MATCH) != 0
    }

    #[inline(always)]
    pub(crate) fn is_dead_state(state: u32) -> bool {
        (state & FLAG_DEAD) != 0
    }

    #[inline(always)]
    pub(crate) fn transition_for_class(&self, state: u32, class_id: usize) -> u32 {
        let state_idx = (state & MASK_STATE) as usize;
        let idx = state_idx * self.class_count as usize + class_id;
        self.transition_table_entry(idx)
    }

    #[inline(always)]
    pub(crate) fn transition_for_byte(&self, state: u32, byte: u8) -> u32 {
        self.transition_for_class(state, self.byte_class(byte))
    }

    #[inline(always)]
    pub(crate) fn transition_for_eoi(&self, state: u32) -> u32 {
        self.transition_for_class(state, self.eoi_class as usize)
    }

    /// Reconstruct a regex DFA from pre-serialized components.
    pub fn from_serialized_parts(
        transition_table: Vec<u32>,
        match_list_pointers: Vec<u32>,
        match_lists: Vec<u32>,
        pattern_lengths: Vec<u32>,
        start_state: u32,
        class_count: u32,
        eoi_class: u32,
        byte_classes: [u32; 256],
        native_dfa_bytes: Vec<u8>,
        native_original_ids: Vec<usize>,
    ) -> Result<Self> {
        if class_count == 0 {
            return Err(Error::PatternCompilationFailed {
                reason: "serialized DFA has class_count=0. Fix: rebuild the compiled pattern set."
                    .to_string(),
            });
        }
        if transition_table.len() % class_count as usize != 0 {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "serialized DFA transition table length {} is not a multiple of class_count {}. Fix: rebuild the compiled pattern set.",
                    transition_table.len(), class_count
                ),
            });
        }
        if eoi_class >= class_count {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "serialized DFA eoi_class {eoi_class} >= class_count {class_count}. Fix: rebuild the compiled pattern set."
                ),
            });
        }
        let (native_dfa, _) = dense::DFA::from_bytes(&native_dfa_bytes).map_err(|error| {
            Error::PatternCompilationFailed {
                reason: format!("serialized regex DFA is invalid: {error}"),
            }
        })?;

        #[cfg(feature = "jit")]
        let jit_dfa = jit_bridge::compile_jit_dfa(
            &transition_table,
            &match_list_pointers,
            &match_lists,
            &pattern_lengths,
            None,
            start_state,
            class_count,
            &byte_classes,
        );

        Ok(Self {
            transition_table,
            match_list_pointers,
            match_lists,
            pattern_lengths,
            start_state,
            class_count,
            eoi_class,
            byte_classes,
            native_dfa: Some(native_dfa.to_owned()),
            native_dfa_bytes,
            native_original_ids,
            transition_table_backing: TransitionTableBacking::Standard,
            fast_regex: None,
            fast_regexes: Vec::new(),
            compact_dfa: None,
            #[cfg(feature = "jit")]
            jit_dfa,
        })
    }

    /// Return the transition table.
    pub fn transition_table(&self) -> &[u32] {
        &self.transition_table
    }
    /// Return the match list pointers.
    pub fn match_list_pointers(&self) -> &[u32] {
        &self.match_list_pointers
    }
    /// Return the match lists.
    pub fn match_lists(&self) -> &[u32] {
        &self.match_lists
    }
    /// Return the pattern lengths.
    pub fn pattern_lengths(&self) -> &[u32] {
        &self.pattern_lengths
    }
    /// Return the start state.
    pub fn start_state(&self) -> u32 {
        self.start_state
    }
    /// Return the number of character classes.
    pub fn class_count(&self) -> u32 {
        self.class_count
    }
    /// Return the end-of-input class.
    pub fn eoi_class(&self) -> u32 {
        self.eoi_class
    }
    /// Return the byte class mappings.
    pub fn byte_classes(&self) -> &[u32; 256] {
        &self.byte_classes
    }

    /// Return the serialized native DFA bytes.
    pub fn native_dfa_bytes(&self) -> &[u8] {
        &self.native_dfa_bytes
    }

    /// Return the regex pattern ID mapping used by the native DFA.
    pub fn native_original_ids(&self) -> &[usize] {
        &self.native_original_ids
    }

    /// Return the cached fast regex (single-pattern only).
    ///
    /// Returns `None` for multi-pattern DFAs or when the regex crate
    /// failed to compile the pattern.
    pub fn fast_regex(&self) -> Option<&regex::bytes::Regex> {
        self.fast_regex.as_ref()
    }

    /// Returns `true` when a JIT DFA is available for this regex backend.
    #[cfg(feature = "jit")]
    #[must_use]
    pub fn has_jit(&self) -> bool {
        self.jit_dfa.is_some()
    }
}

pub(crate) fn alloc_huge_page_vec_with_backing<T>(
    count: usize,
) -> (Vec<T>, TransitionTableBacking) {
    (Vec::with_capacity(count), TransitionTableBacking::Standard)
}

#[cfg(test)]
mod tests {
    use super::RegexDFA;
    use std::mem::{align_of, size_of};

    #[test]
    fn regex_fallback_handles_line_anchors() {
        let dfa = RegexDFA::build(&["^fn main$", "mod"], &[0, 1]).unwrap();
        let mut matches_buf = [crate::Match::from_parts(0, 0, 0); 10];
        let count = dfa
            .scan_native(b"mod demo;\nfn main\nfn main()\n", &mut matches_buf)
            .unwrap();
        let matches = &matches_buf[..count];
        assert!(matches.iter().any(|m| m.pattern_id == 0));
        assert!(matches.iter().any(|m| m.pattern_id == 1));
    }

    #[test]
    fn huge_page_allocator_preserves_vec_semantics() {
        let (mut values, _backing) = super::alloc_huge_page_vec_with_backing::<u32>(1024);
        for value in 0..1024 {
            values.push(value);
        }
        assert_eq!(values.len(), 1024);
        assert_eq!(values[0], 0);
        assert_eq!(values[1023], 1023);
    }

    #[test]
    fn match_struct_is_compact() {
        assert_eq!(align_of::<crate::Match>(), 4); // u32 alignment
        assert_eq!(size_of::<crate::Match>(), 12); // 3 × u32
    }
}
