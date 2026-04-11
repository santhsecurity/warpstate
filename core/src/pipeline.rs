//! Chunking stream orchestration pipeline.
//!
//! Complete mathematical isolation from physical execution engines. Evaluates overlaps,
//! prevents undefined sliding window bounds, and perfectly constructs matching
//! datasets across boundaries mapped abstractly via generic BlockMatcher.

#[cfg(feature = "gpu")]
use crate::{
    dma::{DmaStagingBuffer, FromMappedBuffer},
    error::Result,
    matcher::ZeroCopyBlockMatcher,
};
use crate::{matcher::BlockMatcher, Match, Matcher};

/// Pipeline orchestrator handling memory-stream splitting.
pub struct StreamPipeline<T: BlockMatcher + Send + Sync> {
    backend: T,
    overlap_window: usize,
}

impl<T: BlockMatcher + Send + Sync> StreamPipeline<T> {
    /// Instantiate a deterministic stream splitting pipeline routing to the backend.
    pub fn new(backend: T, overlap_window: usize) -> Self {
        Self {
            backend,
            overlap_window,
        }
    }
}

#[cfg(feature = "gpu")]
impl<T: ZeroCopyBlockMatcher + Send + Sync> StreamPipeline<T> {
    /// Prepare a zero-copy scan session backed by mapped GPU-visible staging memory.
    ///
    /// The caller fills the returned mapped slice, then calls [`ZeroCopyScan::finish`]
    /// to flush the staging buffer and dispatch the backend.
    pub fn scan_zero_copy(&self, input_len: usize) -> Result<ZeroCopyScan<'_, T>> {
        Ok(ZeroCopyScan {
            backend: &self.backend,
            staging: self.backend.create_staging_buffer(input_len)?,
            input_len,
        })
    }
}

/// Two-phase zero-copy scan session.
///
/// ```rust,no_run
/// # async fn example(
/// #     pipeline: &warpstate::StreamPipeline<impl matchkit::BlockMatcher + Send + Sync>,
/// #     data: &[u8],
/// # ) -> warpstate::Result<Vec<warpstate::Match>> {
/// let mut scan = pipeline.scan_zero_copy(data.len())?;
/// scan.mapped_mut().copy_from_slice(data);
/// scan.finish().await
/// # }
/// ```
#[cfg(feature = "gpu")]
pub struct ZeroCopyScan<'a, T: ZeroCopyBlockMatcher + Send + Sync> {
    backend: &'a T,
    staging: DmaStagingBuffer,
    input_len: usize,
}

#[cfg(feature = "gpu")]
impl<T: ZeroCopyBlockMatcher + Send + Sync> ZeroCopyScan<'_, T> {
    /// Returns the mapped writable input region.
    pub fn mapped_mut(&mut self) -> &mut [u8] {
        self.staging.as_mut_slice()
    }

    /// Construct an external buffer adapter directly over the mapped region.
    pub fn external_buffer<B: FromMappedBuffer>(&mut self) -> B {
        B::from_mapped_buffer(self.mapped_mut())
    }

    /// Flush the staging buffer and dispatch the backend.
    pub async fn finish(self) -> Result<Vec<Match>> {
        self.backend
            .scan_zero_copy_block(self.staging, self.input_len)
            .await
    }
}

#[async_trait::async_trait]
impl<T: BlockMatcher + Send + Sync> Matcher for StreamPipeline<T> {
    async fn scan(&self, data: &[u8]) -> matchkit::Result<Vec<Match>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size = self.backend.max_block_size();

        if data.len() <= chunk_size {
            return self.backend.scan_block(data).await;
        }

        // Clamp the overlap window to at most chunk_size - 1, guaranteeing forward
        // progress of at least 1 byte per iteration while maximizing boundary
        // coverage for patterns that span chunk edges.
        let effective_overlap = self.overlap_window.min(chunk_size.saturating_sub(1));

        let mut all_matches = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            let end = std::cmp::min(offset + chunk_size, data.len());
            let chunk_data = &data[offset..end];

            let mut matches = self.backend.scan_block(chunk_data).await?;

            // Adjust to true streaming byte offsets.
            if offset > 0 {
                let offset_u32 =
                    u32::try_from(offset).map_err(|_| matchkit::Error::InputTooLarge {
                        bytes: data.len(),
                        max_bytes: u32::MAX as usize,
                    })?;
                for m in &mut matches {
                    m.start = m.start.saturating_add(offset_u32);
                    m.end = m.end.saturating_add(offset_u32);
                }
            }

            all_matches.append(&mut matches);

            if end == data.len() {
                break;
            }

            let advance = chunk_size.saturating_sub(effective_overlap).max(1);
            offset += advance;
        }

        // De-duplicate matches found in overlapping regions.
        // Sort by (start, pattern_id, end) then merge matches with the same
        // (pattern_id, start) — keep the shortest end. This handles the
        // off-by-one boundary effect where the DFA EOI transition in one
        // chunk adds an extra byte to the match end compared to an inner
        // chunk where the pattern ends mid-stream.
        all_matches.sort_unstable();
        all_matches.dedup_by(|later, earlier| {
            if later.pattern_id == earlier.pattern_id && later.start == earlier.start {
                // Keep the shorter match (smaller end) in `earlier`.
                earlier.end = earlier.end.min(later.end);
                true
            } else {
                false
            }
        });

        Ok(all_matches)
    }
}
