//! Error types for warpstate.

/// Errors that can occur during pattern compilation or matching.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No patterns were added to the builder.
    #[error("pattern set is empty. Fix: add at least one literal pattern before building.")]
    EmptyPatternSet,

    /// A pattern is empty (zero bytes).
    #[error("pattern {index} is empty. Fix: provide a non-empty byte sequence for every pattern.")]
    EmptyPattern {
        /// Index of the empty pattern.
        index: usize,
    },

    /// GPU device could not be acquired.
    #[error("no GPU adapter available. Fix: verify GPU drivers and hardware. For CPU scanning, use warpgrep.")]
    NoGpuAdapter,

    /// GPU device request failed.
    #[error("GPU device request failed: {reason}. Fix: verify your GPU runtime (dx12/vulkan), driver versions, and adapter availability.")]
    GpuDeviceError {
        /// The underlying error message.
        reason: String,
    },

    /// GPU buffer mapping failed.
    #[error("GPU buffer mapping failed. Fix: reduce workload size or retry on a system with stable GPU resources.")]
    BufferMapFailed,

    /// CPU pattern compiler failed to initialize.
    #[error("CPU pattern compilation failed: {reason}. Fix: validate the pattern payloads and retry the scan once the compilation error is resolved.")]
    PatternCompilationFailed {
        /// Underlying compiler error.
        reason: String,
    },

    /// Pattern offsets or lengths do not fit in 32-bit index space.
    #[error("pattern payload is too large for this backend (pattern {index} has {bytes} bytes). Fix: keep each pattern below {max} bytes or split the set into smaller groups.")]
    PatternTooLarge {
        /// Pattern index with the invalid size.
        index: usize,
        /// Pattern size in bytes.
        bytes: usize,
        /// Maximum supported bytes for a single pattern.
        max: usize,
    },

    /// The pattern set does not fit into 32-bit metadata.
    #[error("pattern set metadata is too large ({patterns} patterns, {bytes} total packed bytes). Fix: lower pattern count or trim payload size before compilation.")]
    PatternSetTooLarge {
        /// Number of patterns requested.
        patterns: usize,
        /// Total packed bytes size.
        bytes: usize,
        /// Maximum total packed bytes supported.
        max_bytes: usize,
    },

    /// Input data is larger than the GPU backend supports for one scan.
    #[error("scan input is too large ({bytes} bytes). Fix: shard the input or use CPU scanning for extremely large payloads.")]
    InputTooLarge {
        /// Input size.
        bytes: usize,
        /// Maximum supported bytes.
        max_bytes: usize,
    },

    /// GPU match buffer overflow — too many matches for the configured buffer size.
    #[error("too many matches ({count} exceeds {max}). Fix: reduce pattern count, split input into smaller chunks, or increase MAX_MATCHES if using a custom shader.")]
    MatchBufferOverflow {
        /// Actual number of matches found.
        count: usize,
        /// Maximum matches supported by the buffer.
        max: usize,
    },

    /// A regex pattern contains nested unbounded repetitions that could cause
    /// catastrophic state explosion during DFA compilation (e.g. `(a+)+`).
    #[error("regex pattern {index} contains nested repetitions (pathological). Fix: remove nested quantifiers like `(a+)+` or flatten the expression.")]
    PathologicalRegex {
        /// Index of the offending pattern.
        index: usize,
    },
}

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

// Bridge warpstate errors into matchkit's vocabulary error type.
// This enables `?` in Matcher/BlockMatcher trait impls (which return
// `matchkit::error::Result`) from code that produces `warpstate::Error`.
impl From<Error> for matchkit::Error {
    fn from(error: Error) -> Self {
        match error {
            Error::InputTooLarge { bytes, max_bytes } => Self::InputTooLarge { bytes, max_bytes },
            Error::MatchBufferOverflow { count, max } => Self::MatchBufferOverflow { count, max },
            Error::EmptyPatternSet => Self::EmptyPatternSet,
            Error::EmptyPattern { index } => Self::EmptyPattern { index },
            Error::PatternCompilationFailed { reason } => Self::PatternCompilationFailed { reason },
            other => Self::Backend(Box::new(other)),
        }
    }
}

// Bridge matchkit errors back into warpstate's error type.
impl From<matchkit::Error> for Error {
    fn from(error: matchkit::Error) -> Self {
        match error {
            matchkit::Error::InputTooLarge { bytes, max_bytes } => {
                Self::InputTooLarge { bytes, max_bytes }
            }
            matchkit::Error::MatchBufferOverflow { count, max } => {
                Self::MatchBufferOverflow { count, max }
            }
            matchkit::Error::EmptyPatternSet => Self::EmptyPatternSet,
            matchkit::Error::EmptyPattern { index } => Self::EmptyPattern { index },
            matchkit::Error::PatternCompilationFailed { reason } => {
                Self::PatternCompilationFailed { reason }
            }
            matchkit::Error::Backend(boxed) => Self::GpuDeviceError {
                reason: boxed.to_string(),
            },
            // matchkit::Error is #[non_exhaustive]; future variants map here.
            _ => Self::GpuDeviceError {
                reason: error.to_string(),
            },
        }
    }
}
