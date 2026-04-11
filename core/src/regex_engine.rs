//! Shared byte-regex compilation for CPU fast paths.

use crate::error::{Error, Result};

/// Compile regex syntax with the same byte-oriented assumptions used by DFA
/// construction.
pub(crate) fn build_byte_regex(pattern: &str) -> Result<regex::bytes::Regex> {
    regex::bytes::RegexBuilder::new(pattern)
        .multi_line(true)
        .unicode(false)
        .build()
        .map_err(|error| Error::PatternCompilationFailed {
            reason: format!(
                "regex CPU fast-path compilation failed for `{pattern}`: {error}. Fix: use Rust byte-regex syntax or rebuild with the DFA backend."
            ),
        })
}
