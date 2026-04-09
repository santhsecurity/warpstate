//! Persistent compiled pattern indexes for low-latency pattern loading.

use std::collections::HashSet;
use std::fs;
use std::ops::Range;
use std::path::Path;

use crate::dfa::RegexDFA;
use crate::error::{Error, Result};
use crate::literal_prefilter::LiteralPrefilterTable;

pub mod builder;
pub mod query;

pub(crate) const MAGIC: [u8; 8] = *b"WPSIDX01";
pub(crate) const VERSION: u32 = 3;

/// A serialized on-disk pattern index that can be loaded without recompiling
/// the original pattern set.
#[derive(Debug, Clone)]
pub struct CompiledPatternIndex {
    serialized: Vec<u8>,
    names: Vec<Option<String>>,
    layout: IndexLayout,
}

#[derive(Debug, Clone)]
struct IndexLayout {
    case_insensitive: bool,
    hash_window_len: u32,
    packed_bytes_range: Range<usize>,
    offsets_range: Range<usize>,
    regex_patterns_range: Range<usize>,
    literal_pattern_ids_range: Range<usize>,
    literal_prefilter_range: Range<usize>,
    regex_ranges: Vec<Range<usize>>,
}

#[derive(Debug)]
pub struct ParsedLiterals {
    pub packed_bytes: Vec<u8>,
    pub offsets: Vec<(u32, u32)>,
    pub literal_pattern_ids: Vec<usize>,
    pub literal_prefilter_table: LiteralPrefilterTable,
}

impl CompiledPatternIndex {
    /// Load a compiled index from serialized bytes.
    pub fn load(data: &[u8]) -> Result<Self> {
        let serialized = data.to_vec();
        Self::parse(serialized)
    }

    /// Save this compiled index to a file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        fs::write(path, &self.serialized).map_err(|error| Error::PatternCompilationFailed {
            reason: format!(
                "failed to write compiled index to {}: {error}. Fix: verify the parent directory exists and is writable.",
                path.display()
            ),
        })
    }

    /// Load a compiled index from a file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|error| Error::PatternCompilationFailed {
            reason: format!(
                "failed to read compiled index from {}: {error}. Fix: verify the file exists and is readable.",
                path.display()
            ),
        })?;
        Self::parse(bytes)
    }

    fn parse(serialized: Vec<u8>) -> Result<Self> {
        // Verify CRC32 integrity for version 3 indexes (current version).
        // V3 appends a CRC32 checksum as the last 4 bytes.
        // V1/V2 indexes don't have CRC — skip check for backward compat.
        // Future versions (> VERSION) are rejected after magic+version parse, not here.
        if serialized.len() >= 16 {
            let version_bytes = &serialized[8..12];
            let version = u32::from_le_bytes([
                version_bytes[0],
                version_bytes[1],
                version_bytes[2],
                version_bytes[3],
            ]);
            if version == 3 {
                let payload = &serialized[..serialized.len() - 4];
                let stored_crc = u32::from_le_bytes([
                    serialized[serialized.len() - 4],
                    serialized[serialized.len() - 3],
                    serialized[serialized.len() - 2],
                    serialized[serialized.len() - 1],
                ]);
                let computed = builder::crc32_ieee(payload);
                if stored_crc != computed {
                    return Err(Error::PatternCompilationFailed {
                        reason: format!(
                            "compiled index integrity check failed (CRC mismatch: stored={stored_crc:#010X}, computed={computed:#010X}). \
                             Fix: the index file is corrupted — rebuild it with `warpscan compile`."
                        ),
                    });
                }
            }
        }

        // For v3 indexes, exclude the CRC trailer from the parsing window.
        let parse_end = if serialized.len() >= 16 {
            let v =
                u32::from_le_bytes([serialized[8], serialized[9], serialized[10], serialized[11]]);
            if v == 3 {
                serialized.len() - 4
            } else {
                serialized.len()
            }
        } else {
            serialized.len()
        };
        let mut cursor = Cursor::new(&serialized[..parse_end]);
        let magic = cursor.read_exact(8)?;
        if magic != MAGIC {
            return Err(Error::PatternCompilationFailed {
                reason: "compiled index header is invalid. Fix: rebuild the index file with the current warpstate version.".to_string(),
            });
        }

        let version = cursor.read_u32()?;
        if version > VERSION {
            return Err(Error::PatternCompilationFailed {
                reason: format!(
                    "compiled index version {version} is unsupported. Fix: rebuild the index with this warpstate release."
                ),
            });
        }
        if version == 0 {
            return Err(Error::PatternCompilationFailed {
                reason: "compiled index version is invalid. Fix: rebuild the index file."
                    .to_string(),
            });
        }

        let case_insensitive = cursor.read_u32()? != 0;
        let pattern_count = u32_to_usize(cursor.read_u32()?, "pattern count")?;
        let literal_count = u32_to_usize(cursor.read_u32()?, "literal count")?;
        let regex_count = u32_to_usize(cursor.read_u32()?, "regex dfa count")?;
        let hash_window_len = cursor.read_u32()?;

        let packed_bytes_range = cursor.read_len_prefixed_range()?;
        let packed_bytes = serialized[packed_bytes_range.clone()].to_vec();
        let (offsets_range, offsets) = cursor.read_u32_pairs_range()?;
        if offsets.len() != literal_count {
            return Err(Error::PatternCompilationFailed {
                reason: "compiled index literal metadata is inconsistent. Fix: rebuild the index."
                    .to_string(),
            });
        }
        let names = cursor.read_names()?;
        if names.len() != pattern_count {
            return Err(Error::PatternCompilationFailed {
                reason: "compiled index name metadata is inconsistent. Fix: rebuild the index."
                    .to_string(),
            });
        }

        let regex_patterns_range = if version >= 2 {
            let start = cursor.offset;
            let _ = cursor.read_regex_patterns(pattern_count)?;
            start..cursor.offset
        } else {
            cursor.offset..cursor.offset
        };

        let (literal_pattern_ids_range, literal_pattern_ids) = cursor.read_usize_vec_range()?;
        if literal_pattern_ids.len() != literal_count {
            return Err(Error::PatternCompilationFailed {
                reason: "compiled index literal ID table is inconsistent. Fix: rebuild the index."
                    .to_string(),
            });
        }
        let literal_prefilter_range = cursor.read_prefilter_table_range()?;
        let _ = Cursor::new(&serialized[literal_prefilter_range.clone()]).read_prefilter_table()?;

        for &(start, len) in &offsets {
            let end = start
                .checked_add(len)
                .ok_or_else(|| Error::PatternCompilationFailed {
                    reason: "compiled index literal offset overflows. Fix: rebuild the index."
                        .to_string(),
                })?;
            if end as usize > packed_bytes.len() {
                return Err(Error::PatternCompilationFailed {
                    reason: "compiled index literal offset points outside the packed byte buffer. Fix: rebuild the index."
                        .to_string(),
                });
            }
        }

        let mut regex_ranges = Vec::with_capacity(regex_count);
        for _ in 0..regex_count {
            let regex_start = cursor.offset;
            let transition_table = cursor.read_u32_vec()?;
            let match_list_pointers = cursor.read_u32_vec()?;
            let match_lists = cursor.read_u32_vec()?;
            let pattern_lengths = cursor.read_u32_vec()?;
            let start_state = cursor.read_u32()?;
            let class_count = cursor.read_u32()?;
            let eoi_class = cursor.read_u32()?;
            let byte_classes = cursor.read_u32_array_256()?;
            let native_dfa_bytes = cursor.read_len_prefixed_bytes()?.to_vec();
            let native_original_ids = cursor.read_usize_vec()?;
            let _ = RegexDFA::from_serialized_parts(
                transition_table,
                match_list_pointers,
                match_lists,
                pattern_lengths,
                start_state,
                class_count,
                eoi_class,
                byte_classes,
                native_dfa_bytes,
                native_original_ids,
            )?;
            regex_ranges.push(regex_start..cursor.offset);
        }

        if cursor.remaining() != 0 {
            return Err(Error::PatternCompilationFailed {
                reason: "compiled index contains trailing bytes. Fix: rebuild the index."
                    .to_string(),
            });
        }

        Ok(Self {
            serialized,
            names,
            layout: IndexLayout {
                case_insensitive,
                hash_window_len,
                packed_bytes_range,
                offsets_range,
                regex_patterns_range,
                literal_pattern_ids_range,
                literal_prefilter_range,
                regex_ranges,
            },
        })
    }

    /// Return the optional pattern names in insertion order.
    pub fn names(&self) -> &[Option<String>] {
        &self.names
    }

    /// Rebuild a full PatternSet from the stored index.
    ///
    /// This builds the Aho-Corasick automaton from the stored literal bytes
    /// and loads the serialized regex DFAs. For warpscan's daemon, call this
    /// once at startup then use the PatternSet for all subsequent scans.
    pub fn to_pattern_set(&self) -> Result<crate::PatternSet> {
        if self.layout.regex_patterns_range.is_empty() && !self.layout.regex_ranges.is_empty() {
            return Err(Error::PatternCompilationFailed {
                reason: "compiled index lacks serialized regex patterns. Fix: rebuild index with a newer warpstate that stores regex source."
                    .to_string(),
            });
        }

        let literals = self.parse_literals()?;
        let regex_patterns = self.parse_regex_patterns()?;
        let pattern_count = self.names.len();
        let mut regex_patterns_by_id = vec![None; pattern_count];
        for (pattern_id, pattern) in regex_patterns {
            match regex_patterns_by_id.get_mut(pattern_id) {
                Some(slot) if slot.is_none() => {
                    *slot = Some(pattern);
                }
                Some(_) => {
                    return Err(Error::PatternCompilationFailed {
                        reason:
                            "compiled index contains duplicate regex IDs. Fix: rebuild the index."
                                .to_string(),
                    });
                }
                None => {
                    return Err(Error::PatternCompilationFailed {
                        reason: "compiled index regex pattern ID is out of bounds. Fix: rebuild the index."
                            .to_string(),
                    });
                }
            }
        }

        let mut literal_patterns_by_id = vec![None; pattern_count];
        for (literal_index, pattern_id) in literals.literal_pattern_ids.iter().enumerate() {
            let pattern_id = *pattern_id;
            let &(start, len) = literals.offsets.get(literal_index).ok_or_else(|| {
                Error::PatternCompilationFailed {
                    reason:
                        "compiled index literal metadata is inconsistent. Fix: rebuild the index."
                            .to_string(),
                }
            })?;
            if pattern_id >= pattern_count {
                return Err(Error::PatternCompilationFailed {
                    reason: "compiled index literal pattern ID is out of bounds. Fix: rebuild the index."
                        .to_string(),
                });
            }
            if literal_patterns_by_id[pattern_id].is_some() {
                return Err(Error::PatternCompilationFailed {
                    reason:
                        "compiled index contains duplicate literal IDs. Fix: rebuild the index."
                            .to_string(),
                });
            }
            literal_patterns_by_id[pattern_id] = Some((start, len));
        }

        let mut builder = crate::PatternSet::builder();
        if self.layout.case_insensitive {
            builder = builder.case_insensitive(true);
        }
        for pattern_id in 0..pattern_count {
            if let Some(pattern) = regex_patterns_by_id[pattern_id].take() {
                builder = match self.names[pattern_id].as_ref() {
                    Some(name) => builder.named_regex(name, &pattern),
                    None => builder.regex(&pattern),
                };
                continue;
            }

            let &(start, len) = literal_patterns_by_id[pattern_id].as_ref().ok_or_else(|| {
                Error::PatternCompilationFailed {
                    reason: "compiled index has missing pattern metadata. Fix: rebuild the index."
                        .to_string(),
                }
            })?;
            let bytes = &literals.packed_bytes[start as usize..(start + len) as usize];
            let Some(name) = self.names[pattern_id].as_ref() else {
                builder = builder.literal_bytes(bytes.to_vec());
                continue;
            };
            match std::str::from_utf8(bytes) {
                Ok(literal) => builder = builder.named_literal(name, literal),
                Err(_) => builder = builder.literal_bytes(bytes.to_vec()),
            }
        }
        // Regex patterns are stored as DFA bytes — we can't extract the original
        // regex strings are now stored explicitly and used to rebuild all patterns.
        builder.build()
    }

    /// Return the number of literal patterns in the index.
    pub fn literal_count(&self) -> usize {
        Cursor::new(&self.serialized[self.layout.offsets_range.clone()])
            .read_u32_pairs()
            .map(|offsets| offsets.len())
            .unwrap_or(0)
    }

    /// Parse the literal pattern metadata from the index.
    pub fn parse_literals(&self) -> Result<ParsedLiterals> {
        Ok(ParsedLiterals {
            packed_bytes: self.serialized[self.layout.packed_bytes_range.clone()].to_vec(),
            offsets: Cursor::new(&self.serialized[self.layout.offsets_range.clone()])
                .read_u32_pairs()?,
            literal_pattern_ids: Cursor::new(
                &self.serialized[self.layout.literal_pattern_ids_range.clone()],
            )
            .read_usize_vec()?,
            literal_prefilter_table: Cursor::new(
                &self.serialized[self.layout.literal_prefilter_range.clone()],
            )
            .read_prefilter_table()?,
        })
    }

    /// Parse serialized regex patterns and their original PatternSet IDs.
    pub fn parse_regex_patterns(&self) -> Result<Vec<(usize, String)>> {
        if self.layout.regex_patterns_range.is_empty() {
            return Ok(Vec::new());
        }
        Cursor::new(&self.serialized[self.layout.regex_patterns_range.clone()])
            .read_regex_patterns(self.names.len())
    }

    /// Parse the regex DFA backends from the index.
    pub fn parse_regex_dfas(&self) -> Result<Vec<RegexDFA>> {
        let mut regex_dfas = Vec::with_capacity(self.layout.regex_ranges.len());
        for range in &self.layout.regex_ranges {
            let mut cursor = Cursor::new(&self.serialized[range.clone()]);
            let transition_table = cursor.read_u32_vec()?;
            let match_list_pointers = cursor.read_u32_vec()?;
            let match_lists = cursor.read_u32_vec()?;
            let pattern_lengths = cursor.read_u32_vec()?;
            let start_state = cursor.read_u32()?;
            let class_count = cursor.read_u32()?;
            let eoi_class = cursor.read_u32()?;
            let byte_classes = cursor.read_u32_array_256()?;
            let native_dfa_bytes = cursor.read_len_prefixed_bytes()?.to_vec();
            let native_original_ids = cursor.read_usize_vec()?;
            regex_dfas.push(RegexDFA::from_serialized_parts(
                transition_table,
                match_list_pointers,
                match_lists,
                pattern_lengths,
                start_state,
                class_count,
                eoi_class,
                byte_classes,
                native_dfa_bytes,
                native_original_ids,
            )?);
        }
        Ok(regex_dfas)
    }
}

pub(crate) fn u32_to_usize(value: u32, label: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_|
        Error::PatternCompilationFailed {
            reason: format!("{label} cannot fit in memory on this platform. Fix: rebuild the index on a supported target."),
        }
    )
}

#[derive(Debug)]
pub(crate) struct Cursor<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) offset: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    pub(crate) fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| Error::PatternCompilationFailed {
                reason: "compiled index cursor overflowed. Fix: rebuild the index.".to_string(),
            })?;
        let slice =
            self.data
                .get(self.offset..end)
                .ok_or_else(|| Error::PatternCompilationFailed {
                    reason: "compiled index is truncated. Fix: rebuild the index file.".to_string(),
                })?;
        self.offset = end;
        Ok(slice)
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn read_len_prefixed_bytes(&mut self) -> Result<&'a [u8]> {
        let len = u32_to_usize(self.read_u32()?, "payload length")?;
        self.read_exact(len)
    }

    pub(crate) fn read_len_prefixed_range(&mut self) -> Result<Range<usize>> {
        let len = u32_to_usize(self.read_u32()?, "payload length")?;
        let start = self.offset;
        self.read_exact(len)?;
        Ok(start..start + len)
    }

    pub(crate) fn read_u32_vec(&mut self) -> Result<Vec<u32>> {
        let len = u32_to_usize(self.read_u32()?, "u32 vector length")?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u32()?);
        }
        Ok(values)
    }

    pub(crate) fn read_u32_pairs(&mut self) -> Result<Vec<(u32, u32)>> {
        let len = u32_to_usize(self.read_u32()?, "pair vector length")?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push((self.read_u32()?, self.read_u32()?));
        }
        Ok(values)
    }

    pub(crate) fn read_u32_pairs_range(&mut self) -> Result<(Range<usize>, Vec<(u32, u32)>)> {
        let start = self.offset;
        let values = self.read_u32_pairs()?;
        Ok((start..self.offset, values))
    }

    pub(crate) fn read_usize_vec(&mut self) -> Result<Vec<usize>> {
        let raw = self.read_u32_vec()?;
        raw.into_iter()
            .map(|value| u32_to_usize(value, "usize vector value"))
            .collect()
    }

    pub(crate) fn read_usize_vec_range(&mut self) -> Result<(Range<usize>, Vec<usize>)> {
        let start = self.offset;
        let values = self.read_usize_vec()?;
        Ok((start..self.offset, values))
    }

    pub(crate) fn read_names(&mut self) -> Result<Vec<Option<String>>> {
        let len = u32_to_usize(self.read_u32()?, "name count")?;
        let mut names = Vec::with_capacity(len);
        for _ in 0..len {
            let name_len = self.read_u32()?;
            if name_len == u32::MAX {
                names.push(None);
                continue;
            }
            let raw = self.read_exact(u32_to_usize(name_len, "name length")?)?;
            let name = String::from_utf8(raw.to_vec()).map_err(|error| {
                Error::PatternCompilationFailed {
                    reason: format!(
                        "compiled index name is not valid UTF-8: {error}. Fix: rebuild the index."
                    ),
                }
            })?;
            names.push(Some(name));
        }
        Ok(names)
    }

    pub(crate) fn read_regex_patterns(
        &mut self,
        pattern_count: usize,
    ) -> Result<Vec<(usize, String)>> {
        let len = u32_to_usize(self.read_u32()?, "regex pattern count")?;
        let mut regex_patterns = Vec::with_capacity(len);
        let mut seen = HashSet::new();

        for _ in 0..len {
            let pattern_id = u32_to_usize(self.read_u32()?, "regex pattern id")?;
            if pattern_id >= pattern_count {
                return Err(Error::PatternCompilationFailed {
                    reason:
                        "compiled index regex pattern ID is out of bounds. Fix: rebuild the index."
                            .to_string(),
                });
            }
            if !seen.insert(pattern_id) {
                return Err(Error::PatternCompilationFailed {
                    reason: "compiled index contains duplicate regex pattern IDs. Fix: rebuild the index."
                        .to_string(),
                });
            }
            let pattern = self.read_len_prefixed_bytes()?;
            let pattern = String::from_utf8(pattern.to_vec()).map_err(|error| {
                Error::PatternCompilationFailed {
                    reason: format!(
                        "compiled index regex pattern is not valid UTF-8: {error}. Fix: rebuild the index."
                    ),
                }
            })?;
            regex_patterns.push((pattern_id, pattern));
        }
        Ok(regex_patterns)
    }

    pub(crate) fn read_prefilter_table(&mut self) -> Result<LiteralPrefilterTable> {
        let prefix_meta_len = u32_to_usize(self.read_u32()?, "prefix meta length")?;
        let mut prefix_meta = Vec::with_capacity(prefix_meta_len);
        for _ in 0..prefix_meta_len {
            prefix_meta.push([
                self.read_u32()?,
                self.read_u32()?,
                self.read_u32()?,
                self.read_u32()?,
            ]);
        }

        let bucket_ranges_len = u32_to_usize(self.read_u32()?, "bucket range length")?;
        let mut bucket_ranges = Vec::with_capacity(bucket_ranges_len);
        for _ in 0..bucket_ranges_len {
            bucket_ranges.push([self.read_u32()?, self.read_u32()?]);
        }

        let entries_len = u32_to_usize(self.read_u32()?, "entry length")?;
        let mut entries = Vec::with_capacity(entries_len);
        for _ in 0..entries_len {
            entries.push([self.read_u32()?, self.read_u32()?]);
        }

        Ok(LiteralPrefilterTable {
            prefix_meta,
            bucket_ranges,
            entries,
        })
    }

    pub(crate) fn read_prefilter_table_range(&mut self) -> Result<Range<usize>> {
        let start = self.offset;

        let prefix_meta_len = u32_to_usize(self.read_u32()?, "prefix meta length")?;
        for _ in 0..prefix_meta_len {
            let _ = self.read_u32()?;
            let _ = self.read_u32()?;
            let _ = self.read_u32()?;
            let _ = self.read_u32()?;
        }

        let bucket_ranges_len = u32_to_usize(self.read_u32()?, "bucket range length")?;
        for _ in 0..bucket_ranges_len {
            let _ = self.read_u32()?;
            let _ = self.read_u32()?;
        }

        let entries_len = u32_to_usize(self.read_u32()?, "entry length")?;
        for _ in 0..entries_len {
            let _ = self.read_u32()?;
            let _ = self.read_u32()?;
        }

        Ok(start..self.offset)
    }

    pub(crate) fn read_u32_array_256(&mut self) -> Result<[u32; 256]> {
        let mut values = [0u32; 256];
        for value in &mut values {
            *value = self.read_u32()?;
        }
        Ok(values)
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }
}

#[cfg(test)]
mod adversarial_tests;

#[cfg(test)]
mod tests {
    use super::CompiledPatternIndex;
    use crate::{Error, PatternSet};

    #[test]
    fn round_trips_literal_and_regex_indexes() {
        let patterns = PatternSet::builder()
            .literal("alpha")
            .regex("b[0-9]+")
            .literal("beta")
            .build()
            .unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        let index = CompiledPatternIndex::load(&bytes).unwrap();
        let data = b"alpha b42 beta";
        assert_eq!(index.scan(data).unwrap(), patterns.scan(data).unwrap());
    }

    #[test]
    fn preserves_ascii_case_insensitive_literals() {
        let patterns = PatternSet::builder()
            .case_insensitive(true)
            .literal("Needle")
            .build()
            .unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        let index = CompiledPatternIndex::load(&bytes).unwrap();
        assert_eq!(
            index.scan(b"xxneedlexx").unwrap(),
            patterns.scan(b"xxneedlexx").unwrap()
        );
    }

    #[test]
    fn rejects_invalid_headers() {
        let err = CompiledPatternIndex::load(b"not-an-index").unwrap_err();
        assert!(matches!(err, Error::PatternCompilationFailed { .. }));
    }

    #[test]
    fn saves_and_loads_from_file() {
        let patterns = PatternSet::builder()
            .literal("disk")
            .regex("load")
            .build()
            .unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        let index = CompiledPatternIndex::load(&bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "warpstate-compiled-index-{}.idx",
            std::process::id()
        ));
        index.save_to_file(&path).unwrap();
        let loaded = CompiledPatternIndex::load_from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let data = b"disk load";
        assert_eq!(loaded.scan(data).unwrap(), patterns.scan(data).unwrap());
    }

    #[test]
    fn to_pattern_set_rebuilds_full_pattern_set() {
        let patterns = PatternSet::builder()
            .literal("alpha")
            .named_regex("key", r"[A-Z]{3}-\d+")
            .build()
            .unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        let index = CompiledPatternIndex::load(&bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "warpstate-compiled-index-pattern-set-{}.idx",
            std::process::id()
        ));
        index.save_to_file(&path).unwrap();
        let loaded = CompiledPatternIndex::load_from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let rebuilt = loaded.to_pattern_set().unwrap();
        let data = b"alpha ABC-1234 beta";
        assert_eq!(rebuilt.scan(data).unwrap(), patterns.scan(data).unwrap());
    }

    #[test]
    fn named_patterns_survive_round_trip() {
        let patterns = PatternSet::builder()
            .named_literal("cred", "password")
            .named_regex("key", r"[A-Z]{4}-\d{4}")
            .build()
            .unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        let index = CompiledPatternIndex::load(&bytes).unwrap();
        // Verify scan works (names preserved internally)
        let data = b"password ABCD-1234";
        assert_eq!(index.scan(data).unwrap().len(), 2);
    }

    #[test]
    fn many_literals_round_trip() {
        let mut builder = PatternSet::builder();
        for i in 0..100 {
            builder = builder.literal(&format!("pattern_{i:03}"));
        }
        let patterns = builder.build().unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        let index = CompiledPatternIndex::load(&bytes).unwrap();
        let data = b"xxpattern_042xxpattern_099xx";
        let expected = patterns.scan(data).unwrap();
        let actual = index.scan(data).unwrap();
        assert_eq!(actual.len(), expected.len());
    }

    #[test]
    fn truncated_index_rejected() {
        let patterns = PatternSet::builder().literal("test").build().unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        // Truncate to half
        let err = CompiledPatternIndex::load(&bytes[..bytes.len() / 2]).unwrap_err();
        assert!(matches!(err, Error::PatternCompilationFailed { .. }));
    }

    #[test]
    fn regex_line_anchors_survive_round_trip() {
        let patterns = PatternSet::builder()
            .regex("^fn main$")
            .literal("mod")
            .build()
            .unwrap();
        let bytes = CompiledPatternIndex::build(&patterns).unwrap();
        let index = CompiledPatternIndex::load(&bytes).unwrap();
        let data = b"mod demo;\nfn main\nfn main()\n";
        assert_eq!(index.scan(data).unwrap(), patterns.scan(data).unwrap());
    }
}
