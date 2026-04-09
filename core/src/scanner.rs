//! Synchronous scan abstractions for shared CPU/GPU backends.

use std::sync::Arc;

use crate::{AutoMatcher, HotSwapPatternSet, Match, PatternSet, Result};

/// Byte-oriented scanning interface used by downstream streaming and batch pipelines.
///
/// Implementations must be safe to share across threads when wrapped in [`Arc`].
pub trait ByteScanner: Send + Sync {
    /// Scan `data` and return all matches.
    fn scan_bytes(&self, data: &[u8]) -> Result<Vec<Match>>;
}

impl ByteScanner for PatternSet {
    fn scan_bytes(&self, data: &[u8]) -> Result<Vec<Match>> {
        self.scan(data)
    }
}

impl ByteScanner for HotSwapPatternSet {
    fn scan_bytes(&self, data: &[u8]) -> Result<Vec<Match>> {
        let estimate = crate::specialize::estimate_match_capacity(data.len());
        let mut matches = Vec::with_capacity(estimate);
        self.scan_with(data, |matched| {
            matches.push(matched);
            true
        })?;
        Ok(matches)
    }
}

impl ByteScanner for AutoMatcher {
    fn scan_bytes(&self, data: &[u8]) -> Result<Vec<Match>> {
        self.scan_blocking(data)
    }
}

impl<T> ByteScanner for Arc<T>
where
    T: ByteScanner + ?Sized,
{
    fn scan_bytes(&self, data: &[u8]) -> Result<Vec<Match>> {
        self.as_ref().scan_bytes(data)
    }
}
