//! Streaming scan support built on top of the CPU backend.

use std::sync::Arc;

use crate::cpu;
use crate::{Error, Match, PatternSet, Result};

/// Incremental scanner for chunked input streams.
///
/// The scanner keeps a rolling overlap buffer so matches that span chunk
/// boundaries are still reported exactly once with offsets relative to the
/// full stream.
#[derive(Debug, Clone)]
pub struct StreamScanner {
    patterns: Arc<PatternSet>,
    overlap: Vec<u8>,
    /// Reusable scan buffer — avoids per-feed() allocation.
    window: Vec<u8>,
    max_pattern_len: usize,
    processed_bytes: usize,
}

impl StreamScanner {
    /// Create a streaming scanner from a pattern set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use warpstate::{PatternSet, StreamScanner};
    ///
    /// let patterns = PatternSet::builder().literal("secret").build().unwrap();
    /// let mut scanner = StreamScanner::new(&patterns).unwrap();
    ///
    /// assert!(scanner.feed(b"sec").unwrap().is_empty());
    /// assert_eq!(scanner.feed(b"ret").unwrap().len(), 1);
    /// ```
    pub fn new(patterns: &PatternSet) -> Result<Self> {
        let max_pattern_len = patterns.ir().max_pattern_len;
        if max_pattern_len > u32::MAX as usize {
            return Err(Error::InputTooLarge {
                bytes: max_pattern_len,
                max_bytes: u32::MAX as usize,
            });
        }

        Ok(Self {
            patterns: Arc::new(patterns.clone()),
            overlap: Vec::with_capacity(max_pattern_len),
            window: Vec::new(),
            max_pattern_len,
            processed_bytes: 0,
        })
    }

    /// Feed a chunk of data and return all newly discovered matches.
    ///
    /// Matches are reported with global byte offsets relative to the start of
    /// the stream, not the current chunk.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<Match>> {
        let combined_len = self.overlap.len().saturating_add(chunk.len());
        if combined_len > u32::MAX as usize {
            return Err(Error::InputTooLarge {
                bytes: combined_len,
                max_bytes: u32::MAX as usize,
            });
        }

        let new_processed_bytes = self.processed_bytes.saturating_add(chunk.len());
        if new_processed_bytes > u32::MAX as usize {
            return Err(Error::InputTooLarge {
                bytes: new_processed_bytes,
                max_bytes: u32::MAX as usize,
            });
        }

        let overlap_len = self.overlap.len();
        // Reuse the window buffer: resize to combined length and copy overlap + chunk
        // in a single pass. This avoids clear+reserve+extend overhead by writing
        // directly into the buffer at known offsets.
        self.window.resize(combined_len, 0);
        self.window[..overlap_len].copy_from_slice(&self.overlap);
        self.window[overlap_len..].copy_from_slice(chunk);

        let mut matches = Vec::with_capacity(cpu::estimated_match_capacity(self.window.len()));
        cpu::scan_with(self.patterns.ir(), &self.window, &mut |mat| {
            if matches.len() >= cpu::MAX_CPU_MATCHES {
                return false;
            }
            matches.push(mat);
            true
        })?;
        cpu::sort_matches_if_needed(&mut matches);
        // SAFETY: processed_bytes >= overlap_len because overlap is always filled
        // from previously processed data. saturating_sub is a zero-cost safety net.
        let base_offset = self.processed_bytes.saturating_sub(overlap_len);

        matches.retain(|mat| mat.end as usize > overlap_len);
        for mat in &mut matches {
            mat.start = add_global_offset(mat.start, base_offset)?;
            mat.end = add_global_offset(mat.end, base_offset)?;
        }

        let keep_len = self.max_pattern_len.min(self.window.len());
        // Rotate overlap: move the tail of window into overlap using resize+copy_from_slice.
        // Reuses overlap's existing allocation without intermediate Vec operations.
        let window_len = self.window.len();
        if keep_len == 0 {
            self.overlap.clear();
        } else {
            let start = window_len - keep_len;
            self.overlap.resize(keep_len, 0);
            self.overlap.copy_from_slice(&self.window[start..]);
        }
        self.processed_bytes = new_processed_bytes;

        Ok(matches)
    }

    /// Signal end-of-stream and return any remaining matches.
    ///
    /// This implementation is idempotent: after all available chunks have been
    /// fed, there are no additional complete matches to report beyond the last
    /// `feed` call, so `finish` returns an empty vector.
    pub fn finish(&mut self) -> Result<Vec<Match>> {
        self.feed(&[])
    }
}

fn add_global_offset(offset: u32, base_offset: usize) -> Result<u32> {
    let global_offset = base_offset
        .checked_add(offset as usize)
        .ok_or(Error::InputTooLarge {
            bytes: usize::MAX,
            max_bytes: u32::MAX as usize,
        })?;
    u32::try_from(global_offset).map_err(|_| Error::InputTooLarge {
        bytes: global_offset,
        max_bytes: u32::MAX as usize,
    })
}

#[cfg(test)]
mod tests {
    use crate::{Match, PatternSet, StreamScanner};

    #[test]
    fn finds_matches_spanning_chunk_boundaries() {
        let patterns = PatternSet::builder().literal("abcd").build().unwrap();
        let mut scanner = StreamScanner::new(&patterns).unwrap();

        assert!(scanner.feed(b"ab").unwrap().is_empty());
        let matches = scanner.feed(b"cd").unwrap();

        assert_eq!(
            matches,
            vec![Match {
                pattern_id: 0,
                start: 0,
                end: 4,
                padding: 0,
            }]
        );
        assert!(scanner.finish().unwrap().is_empty());
    }

    #[test]
    fn reports_global_offsets() {
        let patterns = PatternSet::builder().literal("abcd").build().unwrap();
        let mut scanner = StreamScanner::new(&patterns).unwrap();

        assert!(scanner.feed(b"xxab").unwrap().is_empty());
        let matches = scanner.feed(b"cdyy").unwrap();

        assert_eq!(
            matches,
            vec![Match {
                pattern_id: 0,
                start: 2,
                end: 6,
                padding: 0,
            }]
        );
    }

    #[test]
    fn empty_chunks_do_not_break_streaming() {
        let patterns = PatternSet::builder().literal("abc").build().unwrap();
        let mut scanner = StreamScanner::new(&patterns).unwrap();

        assert!(scanner.feed(b"a").unwrap().is_empty());
        assert!(scanner.feed(b"").unwrap().is_empty());
        let matches = scanner.feed(b"bc").unwrap();

        assert_eq!(
            matches,
            vec![Match {
                pattern_id: 0,
                start: 0,
                end: 3,
                padding: 0,
            }]
        );
    }

    #[test]
    fn single_byte_chunks_work() {
        let patterns = PatternSet::builder().literal("needle").build().unwrap();
        let mut scanner = StreamScanner::new(&patterns).unwrap();
        let mut matches = Vec::new();

        for byte in b"needle" {
            matches.extend(scanner.feed(std::slice::from_ref(byte)).unwrap());
        }

        assert_eq!(
            matches,
            vec![Match {
                pattern_id: 0,
                start: 0,
                end: 6,
                padding: 0,
            }]
        );
        assert!(scanner.finish().unwrap().is_empty());
    }
}
