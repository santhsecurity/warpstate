//! Trait definitions for pattern matching backends.
//!
//! The core [`Matcher`] and [`BlockMatcher`] traits are defined in the
//! [`matchkit`] vocabulary crate and re-exported here for convenience.
//! This module adds the GPU-specific [`ZeroCopyBlockMatcher`] extension
//! that cannot live in `matchkit` because it references [`DmaStagingBuffer`].

#[cfg(feature = "gpu")]
use crate::dma::DmaStagingBuffer;

// Re-export vocabulary traits so existing `use crate::matcher::*` works.
pub use matchkit::matcher::BlockMatcher;
#[cfg(feature = "gpu")]
pub use matchkit::matcher::Matcher;

/// Extension trait for backends that can scan directly from mapped staging buffers.
#[cfg(feature = "gpu")]
pub trait ZeroCopyBlockMatcher: BlockMatcher {
    /// Allocate a mapped staging buffer sized for `input_len` bytes of caller-visible input.
    fn create_staging_buffer(&self, input_len: usize) -> crate::error::Result<DmaStagingBuffer>;

    /// Submit a scan using an already-filled mapped staging buffer.
    ///
    /// Implementations must consume the mapped staging buffer, flush it to GPU-visible memory,
    /// and dispatch the backend without copying through an intermediate `Vec<u8>`.
    fn scan_zero_copy_block(
        &self,
        staging: DmaStagingBuffer,
        input_len: usize,
    ) -> impl std::future::Future<Output = crate::error::Result<Vec<matchkit::Match>>> + Send;
}
