use regex_automata::dfa::{dense, Automaton};
use regex_automata::{Anchored, Input};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{RegexDFA, MASK_STATE, MAX_SCAN_MATCHES};
use crate::error::{Error, Result};
use crate::Match;

impl RegexDFA {
    /// Collect matches at a DFA state. Cold path - errors are rare.
    #[cold]
    #[allow(clippy::unused_self)]
    fn collect_fixed_length_matches_at_cold(&self, reason: &'static str) -> Error {
        Error::PatternCompilationFailed {
            reason: format!("{reason}. Fix: rebuild the compiled pattern set."),
        }
    }

    #[inline(always)]
    pub(crate) fn collect_fixed_length_matches_at(
        &self,
        state: u32,
        end: u64,
        matches: &mut Vec<Match>,
        max: usize,
    ) -> Result<()> {
        let state_idx = (state & MASK_STATE) as usize;
        let ptr = self.match_list_pointers[state_idx] as usize;
        let qty = *self.match_lists.get(ptr).ok_or_else(|| {
            self.collect_fixed_length_matches_at_cold(
                "regex DFA match list pointer is out of bounds",
            )
        })? as usize;
        let end_ptr = ptr.checked_add(1 + qty).ok_or_else(|| {
            self.collect_fixed_length_matches_at_cold("regex DFA match list pointer overflowed")
        })?;
        if end_ptr > self.match_lists.len() {
            return Err(self.collect_fixed_length_matches_at_cold(
                "regex DFA match list extends past the serialized table",
            ));
        }
        if matches.len().saturating_add(qty) > max {
            return Err(Error::MatchBufferOverflow { count: max, max });
        }
        matches.reserve(qty);

        for m in 0..qty {
            if matches.len() < max {
                let pat_id = self.match_lists.get(ptr + 1 + m).copied().ok_or_else(|| Error::PatternCompilationFailed {
                    reason: "regex DFA match list pat_id out of bounds. Fix: rebuild the compiled pattern set.".to_string(),
                })?;
                let pat_len = self.pattern_lengths.get(pat_id as usize).copied().ok_or_else(|| Error::PatternCompilationFailed {
                    reason: format!("pattern length out of bounds for pattern {pat_id}. Fix: rebuild the compiled pattern set."),
                })?;
                let start = end.saturating_sub(u64::from(pat_len));
                // u64 to usize cast for error reporting - truncation is ok for error context
                #[allow(clippy::cast_possible_truncation)]
                let start_u32 = u32::try_from(start).map_err(|_| Error::InputTooLarge {
                    bytes: start as usize,
                    max_bytes: u32::MAX as usize,
                })?;
                // u64 to usize cast for error reporting - truncation is ok for error context
                #[allow(clippy::cast_possible_truncation)]
                let end_u32 = u32::try_from(end).map_err(|_| Error::InputTooLarge {
                    bytes: end as usize,
                    max_bytes: u32::MAX as usize,
                })?;
                matches.push(Match {
                    pattern_id: pat_id,
                    start: start_u32,
                    end: end_u32,
                });
            } else {
                return Err(Error::MatchBufferOverflow { count: max, max });
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn scan_suffix_from_state(
        &self,
        haystack: &[u8],
        initial_state: u32,
        absolute_offset: usize,
        matches: &mut Vec<Match>,
        max: usize,
    ) -> Result<u32> {
        if self.pattern_lengths.iter().any(|&len| len == 0) {
            let room = max.saturating_sub(matches.len());
            let mut emitted = 0usize;
            self.scan_native_without_jit_with(haystack, &mut |matched| {
                if emitted >= room {
                    return false;
                }
                let Some(abs_start) = absolute_offset.checked_add(matched.start as usize) else {
                    return false;
                };
                let Some(abs_end) = absolute_offset.checked_add(matched.end as usize) else {
                    return false;
                };
                let Ok(start) = u32::try_from(abs_start) else {
                    return false;
                };
                let Ok(end) = u32::try_from(abs_end) else {
                    return false;
                };
                matches.push(Match {
                    pattern_id: matched.pattern_id,
                    start,
                    end,
                });
                emitted += 1;
                true
            })?;
            if emitted >= room && matches.len() >= max {
                return Err(Error::MatchBufferOverflow { count: max, max });
            }
            let mut state = initial_state;
            for &byte in haystack {
                state = self.transition_for_byte(state, byte);
                if Self::is_dead_state(state) {
                    state = self.start_state;
                }
            }
            return Ok(state);
        }

        let mut state = initial_state;
        let abs_off = absolute_offset as u64;

        // Loop unrolling: process 2 bytes at a time when possible
        let mut pos = 0usize;
        let len = haystack.len();

        // Main unrolled loop
        while pos + 1 < len {
            // Process byte 0
            state = self.transition_for_byte(state, haystack[pos]);

            let end0 = abs_off + pos as u64 + 1;
            if Self::is_match_state(state) && self.match_state_has_only_fixed_lengths(state)? {
                self.collect_fixed_length_matches_at(state, end0, matches, max)?;
            }
            if Self::is_dead_state(state) {
                state = self.start_state;
                state = self.transition_for_byte(state, haystack[pos]);
                if Self::is_match_state(state) && self.match_state_has_only_fixed_lengths(state)? {
                    self.collect_fixed_length_matches_at(state, end0, matches, max)?;
                }
                if Self::is_dead_state(state) {
                    state = self.start_state;
                }
            }
            pos += 1;

            // Process byte 1
            state = self.transition_for_byte(state, haystack[pos]);

            let end1 = abs_off + pos as u64 + 1;
            if Self::is_match_state(state) && self.match_state_has_only_fixed_lengths(state)? {
                self.collect_fixed_length_matches_at(state, end1, matches, max)?;
            }
            if Self::is_dead_state(state) {
                state = self.start_state;
                state = self.transition_for_byte(state, haystack[pos]);
                if Self::is_match_state(state) && self.match_state_has_only_fixed_lengths(state)? {
                    self.collect_fixed_length_matches_at(state, end1, matches, max)?;
                }
                if Self::is_dead_state(state) {
                    state = self.start_state;
                }
            }
            pos += 1;
        }

        // Handle remaining byte(s) — preserving pre-reset state for EOI.
        //
        // When the DFA hits a dead state at the final byte, the reset to
        // start_state would clobber the state needed for EOI-anchored
        // patterns (e.g., `pattern$`). We save the state before reset so
        // the EOI transition below can try both paths.
        let mut pre_reset_state = None;
        while pos < len {
            state = self.transition_for_byte(state, haystack[pos]);

            let end = abs_off + pos as u64 + 1;

            if Self::is_match_state(state) && self.match_state_has_only_fixed_lengths(state)? {
                self.collect_fixed_length_matches_at(state, end, matches, max)?;
            }

            if Self::is_dead_state(state) {
                // Save the dead state before reset — EOI might still match
                // from the pre-transition state at the final byte.
                if pos + 1 == len {
                    pre_reset_state = Some(state);
                }
                state = self.start_state;
                state = self.transition_for_byte(state, haystack[pos]);
                if Self::is_match_state(state) && self.match_state_has_only_fixed_lengths(state)? {
                    self.collect_fixed_length_matches_at(state, end, matches, max)?;
                }
                if Self::is_dead_state(state) {
                    if pos + 1 == len {
                        pre_reset_state = Some(state);
                    }
                    state = self.start_state;
                }
            }
            pos += 1;
        }

        // EOI transition: try both the current state and the pre-reset state
        // to catch matches for patterns anchored to end-of-input (e.g., `$`).
        let eoi_state = self.transition_for_eoi(state);
        if Self::is_match_state(eoi_state) && self.match_state_has_only_fixed_lengths(eoi_state)? {
            // SAFETY: haystack.len() is validated <= u32::MAX by check_input_size
            #[allow(clippy::cast_possible_truncation)]
            let eoi_end = abs_off.saturating_add(haystack.len() as u64);
            self.collect_fixed_length_matches_at(eoi_state, eoi_end, matches, max)?;
        }

        // If the dead-state reset at the final byte clobbered a state that
        // would have produced an EOI match, try the pre-reset state's EOI.
        if let Some(pre_state) = pre_reset_state {
            let eoi_pre = self.transition_for_eoi(pre_state);
            if Self::is_match_state(eoi_pre)
                && self.match_state_has_only_fixed_lengths(eoi_pre)?
                && eoi_pre != eoi_state
            {
                #[allow(clippy::cast_possible_truncation)]
                let eoi_end = abs_off.saturating_add(haystack.len() as u64);
                self.collect_fixed_length_matches_at(eoi_pre, eoi_end, matches, max)?;
            }
        }

        Ok(state)
    }

    fn match_state_has_only_fixed_lengths(&self, state: u32) -> Result<bool> {
        let state_idx = (state & MASK_STATE) as usize;
        let ptr = self
            .match_list_pointers
            .get(state_idx)
            .copied()
            .ok_or_else(|| {
                self.collect_fixed_length_matches_at_cold(
                    "regex DFA match list pointer is out of bounds",
                )
            })? as usize;
        let qty = *self.match_lists.get(ptr).ok_or_else(|| {
            self.collect_fixed_length_matches_at_cold(
                "regex DFA match list pointer is out of bounds",
            )
        })? as usize;
        for m in 0..qty {
            let pat_id = self.match_lists.get(ptr + 1 + m).copied().ok_or_else(|| {
                self.collect_fixed_length_matches_at_cold(
                    "regex DFA match list extends past the serialized table",
                )
            })? as usize;
            let pat_len = self.pattern_lengths.get(pat_id).copied().ok_or_else(|| {
                self.collect_fixed_length_matches_at_cold(
                    "regex DFA pattern length lookup out of bounds",
                )
            })?;
            if pat_len == 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Single-pass O(n) DFA scan.
    ///
    /// Walks the DFA once left-to-right across the entire haystack. At each byte
    /// position it transitions the current state and, if a match (or dead) state
    /// is reached, records matches and resets to the start state so the next
    /// potential match can begin. After all input bytes are consumed, a final
    /// EOI transition is attempted to catch anchored-end patterns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MatchBufferOverflow`] when the internal match limit is
    /// reached.
    #[inline]
    pub fn scan(&self, haystack: &[u8], out_matches: &mut [Match]) -> Result<usize> {
        self.scan_without_jit(haystack, out_matches)
    }

    /// Interpreted DFA scan used when JIT is unavailable or for parity testing.
    #[inline]
    pub fn scan_without_jit(&self, haystack: &[u8], out_matches: &mut [Match]) -> Result<usize> {
        let mut count = 0usize;
        self.scan_native_without_jit_with(haystack, &mut |matched| {
            if count >= out_matches.len() {
                false
            } else {
                out_matches[count] = matched;
                count += 1;
                true
            }
        })?;
        Ok(count)
    }

    /// Cold path for match collection errors.
    #[cold]
    #[allow(clippy::unused_self)]
    fn collect_matches_cold(&self, msg: &'static str) -> Error {
        Error::PatternCompilationFailed {
            reason: format!("{msg}. Fix: rebuild the compiled pattern set."),
        }
    }

    /// Helper to emit all pattern matches at a given DFA accept state.
    #[inline(always)]
    fn collect_matches_at(
        &self,
        state: u32,
        start: u64,
        end: u64,
        matches: &mut Vec<Match>,
        max: usize,
    ) -> Result<()> {
        let state_idx = (state & MASK_STATE) as usize;
        let ptr = self
            .match_list_pointers
            .get(state_idx)
            .copied()
            .ok_or_else(|| self.collect_matches_cold("match list pointers out of bounds"))?
            as usize;
        let qty = self
            .match_lists
            .get(ptr)
            .copied()
            .ok_or_else(|| self.collect_matches_cold("match lists out of bounds"))?
            as usize;
        for m in 0..qty {
            if matches.len() >= max {
                return Err(Error::MatchBufferOverflow { count: max, max });
            }
            let pat_id = self
                .match_lists
                .get(ptr + 1 + m)
                .copied()
                .ok_or_else(|| self.collect_matches_cold("match lists out of bounds"))?;
            // u64 to usize cast for error reporting - truncation is ok for error context
            #[allow(clippy::cast_possible_truncation)]
            let start_u32 = u32::try_from(start).map_err(|_| Error::InputTooLarge {
                bytes: start as usize,
                max_bytes: u32::MAX as usize,
            })?;
            // u64 to usize cast for error reporting - truncation is ok for error context
            #[allow(clippy::cast_possible_truncation)]
            let end_u32 = u32::try_from(end).map_err(|_| Error::InputTooLarge {
                bytes: end as usize,
                max_bytes: u32::MAX as usize,
            })?;
            matches.push(Match {
                pattern_id: pat_id,
                start: start_u32,
                end: end_u32,
            });
        }
        Ok(())
    }

    /// CPU-correct regex scan using `regex-automata`'s native search.
    ///
    /// Uses unanchored forward search (O(n)) to find match ends, then
    /// walks backward with anchored search to find starts. The custom
    /// transition table is still available for GPU backends.
    #[inline]
    pub fn scan_native(&self, haystack: &[u8], out_matches: &mut [Match]) -> Result<usize> {
        let mut count = 0;
        self.scan_native_with(haystack, &mut |matched| {
            if count >= out_matches.len() {
                false
            } else {
                out_matches[count] = matched;
                count += 1;
                true
            }
        })?;
        Ok(count)
    }

    /// CPU-correct regex scan that bypasses JIT even when the feature is enabled.
    #[inline]
    pub fn scan_native_without_jit(
        &self,
        haystack: &[u8],
        out_matches: &mut [Match],
    ) -> Result<usize> {
        let mut count = 0;
        self.scan_native_without_jit_with(haystack, &mut |matched| {
            if count >= out_matches.len() {
                false
            } else {
                out_matches[count] = matched;
                count += 1;
                true
            }
        })?;
        Ok(count)
    }

    /// CPU-correct regex scan that streams matches into `visitor`.
    #[inline]
    pub fn scan_native_with<F>(&self, haystack: &[u8], visitor: &mut F) -> Result<()>
    where
        F: FnMut(Match) -> bool,
    {
        // JIT DFA disabled for scan_native_with — the fast_regex path in
        // scan_native_without_jit_with is faster and more correct for single-pattern
        // DFAs. JIT is still used by scan() for the buffer-based API.

        self.scan_native_without_jit_with(haystack, visitor)
    }

    #[inline]
    fn scan_native_without_jit_with<F>(&self, haystack: &[u8], visitor: &mut F) -> Result<()>
    where
        F: FnMut(Match) -> bool,
    {
        // Fast path: use regex crate's optimized engine (SIMD + lazy DFA)
        // instead of byte-by-byte dense DFA walk with backwards start search.
        // Only safe for single-pattern DFAs — multi-pattern combined alternation
        // loses pattern identity and changes match priority semantics.
        if self.native_original_ids.len() == 1 {
            if let Some(fast_regex) = &self.fast_regex {
                let pat_id = self.native_original_ids[0] as u32;
                for m in fast_regex.find_iter(haystack) {
                    // SAFETY: haystack length is validated <= u32::MAX by check_input_size
                    // at all call sites. These conversions cannot fail.
                    let start = u32::try_from(m.start()).map_err(|_| Error::InputTooLarge {
                        bytes: m.start(),
                        max_bytes: u32::MAX as usize,
                    })?;
                    let end = u32::try_from(m.end()).map_err(|_| Error::InputTooLarge {
                        bytes: m.end(),
                        max_bytes: u32::MAX as usize,
                    })?;
                    if !visitor(Match {
                        pattern_id: pat_id,
                        start,
                        end,
                    }) {
                        break;
                    }
                }
                return Ok(());
            }
        }

        // Multi-pattern fast path: per-pattern regex crate engines with k-way merge.
        // Each regex uses SIMD + lazy DFA, producing (start, end) in one pass.
        // K-way merge emits matches in byte-offset order. ~3x faster than dense
        // DFA + O(256) backwards start search per match.
        if !self.fast_regexes.is_empty()
            && self.fast_regexes.len() == self.native_original_ids.len()
        {
            return Self::scan_multi_regex_kway(&self.fast_regexes, haystack, visitor);
        }

        if let Some(compact) = &self.compact_dfa {
            return Self::scan_native_with_compact(
                compact,
                &self.pattern_lengths,
                haystack,
                visitor,
            );
        }

        if let Some(native_dfa) = &self.native_dfa {
            return Self::scan_native_with_dfa(
                native_dfa,
                &self.native_original_ids,
                &self.pattern_lengths,
                haystack,
                visitor,
            );
        }
        let (native_dfa, _) = dense::DFA::from_bytes(&self.native_dfa_bytes).map_err(|error| {
            Error::PatternCompilationFailed {
                reason: format!("serialized regex DFA is invalid: {error}"),
            }
        })?;
        Self::scan_native_with_dfa(
            &native_dfa,
            &self.native_original_ids,
            &self.pattern_lengths,
            haystack,
            visitor,
        )
    }

    #[inline]
    fn scan_native_with_compact<F>(
        compact: &super::CompactDfa,
        pattern_lengths: &[u32],
        haystack: &[u8],
        visitor: &mut F,
    ) -> Result<()>
    where
        F: FnMut(Match) -> bool,
    {
        let mut pos = 0usize;
        let mut count = 0usize;

        while pos < haystack.len() {
            let mut at = pos;
            let mut state = compact.start_unanchored;
            let mut match_end = None;
            let mut match_pat = 0;

            while at < haystack.len() {
                let byte = haystack[at];
                state = compact.next_state(state, byte);

                if compact.is_match_state(state) {
                    match_end = Some(at + 1);
                    match_pat = compact.match_pattern[state as usize];
                }
                if compact.is_dead_state(state) || compact.is_quit_state(state) {
                    break;
                }
                at += 1;
            }

            if match_end.is_none() && !compact.is_dead_state(state) && !compact.is_quit_state(state)
            {
                let eoi_state = compact.next_eoi_state(state);
                if compact.is_match_state(eoi_state) {
                    match_end = Some(haystack.len());
                    match_pat = compact.match_pattern[eoi_state as usize];
                }
            }

            let Some(end) = match_end else {
                break;
            };

            let max_pat_len = pattern_lengths
                .get(match_pat as usize)
                .copied()
                .unwrap_or(0) as usize;
            let match_start = if max_pat_len > 0 && end >= max_pat_len {
                end - max_pat_len
            } else {
                let search_start = pos;
                let mut best = end.saturating_sub(1).max(pos);
                for candidate in (search_start..end).rev() {
                    let mut b_state = compact.start_anchored;
                    let mut b_at = candidate;
                    let mut b_match = false;
                    while b_at < end {
                        b_state = compact.next_state(b_state, haystack[b_at]);
                        if compact.is_match_state(b_state) && b_at + 1 == end {
                            b_match = true;
                        }
                        if compact.is_dead_state(b_state) || compact.is_quit_state(b_state) {
                            break;
                        }
                        b_at += 1;
                    }
                    if b_at == end && !b_match {
                        let eoi_state = compact.next_eoi_state(b_state);
                        if compact.is_match_state(eoi_state) {
                            b_match = true;
                        }
                    }
                    if b_match {
                        best = candidate;
                    }
                }
                best
            };

            if count >= super::MAX_SCAN_MATCHES {
                return Err(Error::MatchBufferOverflow {
                    count: super::MAX_SCAN_MATCHES,
                    max: super::MAX_SCAN_MATCHES,
                });
            }

            // SAFETY: match indices are bounded by haystack.len() <= u32::MAX
            #[allow(clippy::cast_possible_truncation)]
            let pattern_id = match_pat;
            #[allow(clippy::cast_possible_truncation)]
            let start = match_start as u32;
            #[allow(clippy::cast_possible_truncation)]
            let end_u32 = end as u32;
            if !visitor(Match {
                pattern_id,
                start,
                end: end_u32,
            }) {
                break;
            }
            count += 1;
            pos = end.max(pos + 1);
        }

        Ok(())
    }

    /// Multi-pattern scan using per-pattern regex::bytes::Regex with k-way merge.
    /// Each regex uses the regex crate's optimized engine (SIMD prefilter + lazy DFA)
    /// which gives (start, end) in one pass — no backwards search needed.
    ///
    /// Emits ALL matches at the minimum start position before advancing,
    /// so multiple patterns matching at the same byte offset are all reported.
    fn scan_multi_regex_kway<F>(
        fast_regexes: &[(regex::bytes::Regex, u32)],
        haystack: &[u8],
        visitor: &mut F,
    ) -> Result<()>
    where
        F: FnMut(Match) -> bool,
    {
        struct StreamState<'a> {
            iter: regex::bytes::Matches<'a, 'a>,
            pattern_id: u32,
            next: Option<regex::bytes::Match<'a>>,
        }

        let mut streams: Vec<StreamState<'_>> = fast_regexes
            .iter()
            .map(|(re, pat_id)| {
                let mut iter = re.find_iter(haystack);
                let next = iter.next();
                StreamState {
                    iter,
                    pattern_id: *pat_id,
                    next,
                }
            })
            .collect();
        let mut heap = BinaryHeap::new();
        for (stream_index, stream) in streams.iter().enumerate() {
            if let Some(next) = &stream.next {
                heap.push(Reverse((
                    next.start(),
                    stream.pattern_id,
                    next.end(),
                    stream_index,
                )));
            }
        }

        let mut count = 0usize;
        while let Some(Reverse((start_usize, _pattern_key, end_usize, stream_index))) = heap.pop() {
            let s = &mut streams[stream_index];
            let Some(m) = s.next.take() else { continue };
            if m.start() != start_usize || m.end() != end_usize {
                continue;
            }
            if count >= MAX_SCAN_MATCHES {
                return Err(Error::MatchBufferOverflow {
                    count: MAX_SCAN_MATCHES,
                    max: MAX_SCAN_MATCHES,
                });
            }
            // SAFETY: match indices are bounded by haystack.len() <= u32::MAX
            #[allow(clippy::cast_possible_truncation)]
            let start = start_usize as u32;
            #[allow(clippy::cast_possible_truncation)]
            let end = end_usize as u32;
            let mat = Match {
                pattern_id: s.pattern_id,
                start,
                end,
            };
            s.next = s.iter.next();
            if let Some(next) = &s.next {
                heap.push(Reverse((
                    next.start(),
                    s.pattern_id,
                    next.end(),
                    stream_index,
                )));
            }
            if !visitor(mat) {
                return Ok(());
            }
            count += 1;
        }
        Ok(())
    }

    fn scan_native_with_dfa<T, F>(
        native_dfa: &dense::DFA<T>,
        native_original_ids: &[usize],
        pattern_lengths: &[u32],
        haystack: &[u8],
        visitor: &mut F,
    ) -> Result<()>
    where
        T: AsRef<[u32]>,
        F: FnMut(Match) -> bool,
    {
        let mut pos = 0usize;
        let mut count = 0usize;

        while pos < haystack.len() {
            // Use earliest(false) to get leftmost-longest semantics matching
            // regex::find_iter behavior. earliest(true) produces shortest
            // matches which diverges from PatternSet::scan.
            let input = Input::new(haystack)
                .range(pos..)
                .anchored(Anchored::No)
                .earliest(false);
            let result =
                native_dfa
                    .try_search_fwd(&input)
                    .map_err(|e| Error::PatternCompilationFailed {
                        reason: format!("DFA search failed: {e}"),
                    })?;

            let Some(half_match) = result else {
                break;
            };

            let match_end = half_match.offset();
            let pat_idx = half_match.pattern().as_usize();
            let pat_id = native_original_ids.get(pat_idx).copied().ok_or_else(|| Error::PatternCompilationFailed {
                reason: format!("pattern ID lookup failed: pat_idx {} >= native_original_ids.len() ({}). Fix: GPU/CPU state mismatch or corrupted DFA state.", pat_idx, native_original_ids.len()),
            })?;
            let max_pat_len = pattern_lengths.get(pat_id).copied().unwrap_or(0) as usize;
            let match_start = if max_pat_len > 0 && match_end >= max_pat_len {
                match_end - max_pat_len
            } else {
                let search_start = pos;
                let mut best = match_end.saturating_sub(1).max(pos);
                for candidate in (search_start..match_end).rev() {
                    let input_a = Input::new(haystack)
                        .range(candidate..)
                        .anchored(Anchored::Yes);
                    if let Ok(Some(hm)) = native_dfa.try_search_fwd(&input_a) {
                        if hm.offset() == match_end {
                            best = candidate;
                        }
                    }
                }
                best
            };

            if count >= MAX_SCAN_MATCHES {
                return Err(Error::MatchBufferOverflow {
                    count: MAX_SCAN_MATCHES,
                    max: MAX_SCAN_MATCHES,
                });
            }

            // SAFETY: match indices are bounded by haystack.len() <= u32::MAX
            #[allow(clippy::cast_possible_truncation)]
            let pattern_id = pat_id as u32;
            #[allow(clippy::cast_possible_truncation)]
            let start = match_start as u32;
            #[allow(clippy::cast_possible_truncation)]
            let end = match_end as u32;
            if !visitor(Match {
                pattern_id,
                start,
                end,
            }) {
                break;
            }
            count += 1;
            pos = match_end.max(pos + 1);
        }

        Ok(())
    }
}
