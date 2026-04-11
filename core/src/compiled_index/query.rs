use crate::compiled_index::{CompiledPatternIndex, ParsedLiterals};
use crate::error::{Error, Result};
use crate::literal_prefilter::LiteralPrefilterTable;
use crate::pattern::literal_prefilter_hash;
use crate::Match;

impl CompiledPatternIndex {
    /// Scan a payload using the pre-compiled index.
    ///
    /// **Performance note**: This uses the index's built-in hash-table scanner
    /// which is O(n × pattern_count) for case-insensitive scans. For repeated
    /// scanning (daemon, streaming), call [`to_pattern_set()`](Self::to_pattern_set)
    /// once and use the returned `PatternSet` — it builds an Aho-Corasick
    /// automaton for O(n) scanning regardless of pattern count.
    pub fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        if data.len() > u32::MAX as usize {
            return Err(Error::InputTooLarge {
                bytes: data.len(),
                max_bytes: u32::MAX as usize,
            });
        }

        let mut matches = if self.literal_count() == 1
            && self.layout.regex_ranges.is_empty()
            && !self.layout.case_insensitive
        {
            let literals = self.parse_literals()?;
            self.scan_single_literal(data, &literals)?
        } else {
            let literals = self.parse_literals()?;
            self.scan_literals(data, &literals)?
        };
        for dfa in self.parse_regex_dfas()? {
            let mut dfa_matches = vec![Match::from_parts(0, 0, 0); 1024];
            let n = dfa.scan_native(data, &mut dfa_matches)?;
            matches.extend(dfa_matches.into_iter().take(n));
        }
        sort_matches_if_needed(&mut matches);
        finish_matches(matches)
    }

    #[allow(clippy::unused_self)]
    fn scan_single_literal(&self, data: &[u8], literals: &ParsedLiterals) -> Result<Vec<Match>> {
        let (start, len) = literals.offsets[0];
        let needle = &literals.packed_bytes[start as usize..(start + len) as usize];
        let finder = memchr::memmem::Finder::new(needle);
        let mut matches = Vec::new();
        // pattern IDs are bounded by the number of patterns in the set
        #[allow(clippy::cast_possible_truncation)]
        let pattern_id = literals.literal_pattern_ids[0] as u32;
        for pos in finder.find_iter(data) {
            // SAFETY: pos < data.len() which is validated <= u32::MAX at function entry
            #[allow(clippy::cast_possible_truncation)]
            let start = pos as u32;
            // SAFETY: pos < data.len() and len is bounded, so pos + len <= data.len() <= u32::MAX
            #[allow(clippy::cast_possible_truncation)]
            let end = pos as u32 + len;
            matches.push(Match {
                pattern_id,
                start,
                end,
            });
        }
        Ok(matches)
    }

    #[allow(clippy::unused_self)]
    fn scan_literals(&self, data: &[u8], literals: &ParsedLiterals) -> Result<Vec<Match>> {
        if self.layout.case_insensitive {
            return self.scan_literals_case_insensitive(data, literals);
        }

        let mut matches = Vec::new();
        let mut pos = 0usize;

        while pos < data.len() {
            let mut best: Option<Match> = None;
            let max_prefix = (data.len() - pos).min(self.layout.hash_window_len.max(1) as usize);

            for prefix_len in 1..=max_prefix {
                let hash = literal_prefilter_hash(&data[pos..], prefix_len);
                // prefix_len is bounded by hash_window_len which is a small constant
                #[allow(clippy::cast_possible_truncation)]
                for literal_index in self.prefilter_candidates(literals, prefix_len as u32, hash) {
                    let literal_index = literal_index as usize;
                    let (start, len) = literals.offsets[literal_index];
                    let len = len as usize;
                    if pos + len > data.len() {
                        continue;
                    }
                    let needle = &literals.packed_bytes[start as usize..(start as usize + len)];
                    if bytes_match(needle, &data[pos..pos + len], self.layout.case_insensitive) {
                        // pattern IDs are bounded by the number of patterns in the set
                        #[allow(clippy::cast_possible_truncation)]
                        let pattern_id = literals.literal_pattern_ids[literal_index] as u32;
                        let candidate = Match {
                            pattern_id,
                            // SAFETY: pos < data.len() which is validated <= u32::MAX at function entry
                            start: pos as u32,
                            // SAFETY: pos + len <= data.len() which is validated <= u32::MAX
                            end: (pos + len) as u32,
                        };
                        if best
                            .as_ref()
                            .map_or(true, |current| match_precedes(&candidate, current))
                        {
                            best = Some(candidate);
                        }
                    }
                }
            }

            if let Some(found) = best {
                pos = found.end as usize;
                matches.push(found);
            } else {
                pos += 1;
            }
        }

        Ok(matches)
    }

    #[allow(clippy::unused_self)]
    fn scan_literals_case_insensitive(
        &self,
        data: &[u8],
        literals: &ParsedLiterals,
    ) -> Result<Vec<Match>> {
        let mut matches = Vec::new();
        let mut pos = 0usize;

        while pos < data.len() {
            let mut best: Option<Match> = None;
            for (literal_index, &(start, len)) in literals.offsets.iter().enumerate() {
                let len = len as usize;
                if pos + len > data.len() {
                    continue;
                }
                // SAFETY: start is bounded by packed_bytes.len() which fits in usize
                #[allow(clippy::cast_possible_truncation)]
                let needle = &literals.packed_bytes[start as usize..(start as usize + len)];
                if bytes_match(needle, &data[pos..pos + len], true) {
                    // pattern IDs are bounded by the number of patterns in the set
                    #[allow(clippy::cast_possible_truncation)]
                    let pattern_id = literals.literal_pattern_ids[literal_index] as u32;
                    // SAFETY: pos < data.len() which is validated <= u32::MAX at function entry
                    #[allow(clippy::cast_possible_truncation)]
                    let start = pos as u32;
                    // SAFETY: pos + len <= data.len() which is validated <= u32::MAX
                    #[allow(clippy::cast_possible_truncation)]
                    let end = (pos + len) as u32;
                    let candidate = Match {
                        pattern_id,
                        start,
                        end,
                    };
                    if best
                        .as_ref()
                        .map_or(true, |current| match_precedes(&candidate, current))
                    {
                        best = Some(candidate);
                    }
                }
            }

            if let Some(found) = best {
                pos = found.end as usize;
                matches.push(found);
            } else {
                pos += 1;
            }
        }

        Ok(matches)
    }

    #[allow(clippy::unused_self)]
    fn prefilter_candidates<'a>(
        &'a self,
        literals: &'a ParsedLiterals,
        prefix_len: u32,
        hash: u32,
    ) -> impl Iterator<Item = u32> + 'a {
        let Some(meta) = prefix_len.checked_sub(1).and_then(|idx| {
            literals
                .literal_prefilter_table
                .prefix_meta
                .get(idx as usize)
        }) else {
            return EitherIter::Empty(std::iter::empty());
        };

        if meta[2] == 0 {
            return EitherIter::Empty(std::iter::empty());
        }

        let bucket_index = meta[0] + (hash & meta[1]);
        let Some(range) = self
            .literal_prefilter_table(literals)
            .bucket_ranges
            .get(bucket_index as usize)
            .copied()
        else {
            return EitherIter::Empty(std::iter::empty());
        };
        let Ok(start) = usize::try_from(range[0]) else {
            return EitherIter::Empty(std::iter::empty());
        };
        let Ok(len) = usize::try_from(range[1]) else {
            return EitherIter::Empty(std::iter::empty());
        };
        let Some(end) = start.checked_add(len) else {
            return EitherIter::Empty(std::iter::empty());
        };
        let Some(entries) = literals.literal_prefilter_table.entries.get(start..end) else {
            return EitherIter::Empty(std::iter::empty());
        };
        EitherIter::Slice(
            entries
                .iter()
                .filter(move |entry| entry[0] == hash)
                .map(|entry| entry[1]),
        )
    }

    #[allow(clippy::unused_self)] // Method for future extensibility
    fn literal_prefilter_table<'a>(
        &self,
        literals: &'a ParsedLiterals,
    ) -> &'a LiteralPrefilterTable {
        &literals.literal_prefilter_table
    }
}

fn bytes_match(expected: &[u8], actual: &[u8], case_insensitive: bool) -> bool {
    if !case_insensitive {
        return expected == actual;
    }

    expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(&left, &right)| left.eq_ignore_ascii_case(&right))
}

/// Leftmost-longest match priority matching Aho-Corasick's `LeftmostFirst`
/// semantics used by `PatternSet::scan()`.
///
/// AC `LeftmostFirst` continues matching after a short pattern completes,
/// reporting the **longest** match at each start position. For patterns
/// `["p", "p6"]` at input `"p6"`, AC reports `"p6"` (end=36) not `"p"`
/// (end=35) because the automaton tentatively matches `"p"` but extends
/// to `"p6"` before emitting.
///
/// Priority: longest match first (higher end), then lowest pattern_id.
fn match_precedes(left: &Match, right: &Match) -> bool {
    left.end > right.end || (left.end == right.end && left.pattern_id < right.pattern_id)
}

fn sort_matches_if_needed(matches: &mut [Match]) {
    if matches.windows(2).all(|w| w[0] <= w[1]) {
        return;
    }
    matches.sort_unstable();
}

fn finish_matches(matches: Vec<Match>) -> Result<Vec<Match>> {
    if matches.len() > 1_048_576 {
        return Err(Error::MatchBufferOverflow {
            count: matches.len(),
            max: 1_048_576,
        });
    }
    Ok(matches)
}

enum EitherIter<L, R> {
    Empty(L),
    Slice(R),
}

impl<L, R, T> Iterator for EitherIter<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty(iter) => iter.next(),
            Self::Slice(iter) => iter.next(),
        }
    }
}
