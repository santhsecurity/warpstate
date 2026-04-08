//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::{CompiledPatternIndex, MAGIC, VERSION};
use crate::Error;
use crate::PatternSet;

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

#[cfg(test)]
pub mod load_wrong_magic_bytes_errors;
#[cfg(test)]
pub use load_wrong_magic_bytes_errors::*;
pub mod load_future_version_errors;
pub use load_future_version_errors::*;
#[cfg(test)]
pub mod load_version_zero_errors;
#[cfg(test)]
pub use load_version_zero_errors::*;
#[cfg(test)]
pub mod load_max_version_errors;
#[cfg(test)]
pub use load_max_version_errors::*;
pub mod load_truncated_at_various_boundaries;
pub use load_truncated_at_various_boundaries::*;
#[cfg(test)]
pub mod load_truncated_header_fields;
#[cfg(test)]
pub use load_truncated_header_fields::*;
pub mod load_literal_count_mismatch_detected;
pub use load_literal_count_mismatch_detected::*;
#[cfg(test)]
pub mod load_literal_count_zero_with_offsets;
#[cfg(test)]
pub use load_literal_count_zero_with_offsets::*;
pub mod load_offset_past_packed_bytes_detected;
pub use load_offset_past_packed_bytes_detected::*;
#[cfg(test)]
pub mod load_offset_overflow_detected;
#[cfg(test)]
pub use load_offset_overflow_detected::*;
pub mod round_trip_literal_and_regex_results_identical;
pub use round_trip_literal_and_regex_results_identical::*;
#[cfg(test)]
pub mod round_trip_case_insensitive_results_identical;
#[cfg(test)]
pub use round_trip_case_insensitive_results_identical::*;
#[cfg(test)]
pub mod proptest_roundtrip_preserves_scan_results;
#[cfg(test)]
pub use proptest_roundtrip_preserves_scan_results::*;
pub mod empty_pattern_set_errors_on_build;
pub use empty_pattern_set_errors_on_build::*;
pub mod large_pattern_set_100k_does_not_oom;
pub use large_pattern_set_100k_does_not_oom::*;
#[cfg(test)]
pub mod medium_pattern_set_10k_roundtrip;
#[cfg(test)]
pub use medium_pattern_set_10k_roundtrip::*;
pub mod null_bytes_pattern_roundtrip;
pub use null_bytes_pattern_roundtrip::*;
#[cfg(test)]
pub mod multiple_patterns_with_null_bytes;
#[cfg(test)]
pub use multiple_patterns_with_null_bytes::*;
#[cfg(test)]
pub mod pattern_starting_with_null;
#[cfg(test)]
pub use pattern_starting_with_null::*;
#[cfg(test)]
pub mod pattern_ending_with_null;
#[cfg(test)]
pub use pattern_ending_with_null::*;
pub mod pattern_count_mismatch_detected;
pub use pattern_count_mismatch_detected::*;
#[cfg(test)]
pub mod names_count_mismatch_detected;
#[cfg(test)]
pub use names_count_mismatch_detected::*;
#[cfg(test)]
pub mod to_pattern_set_rebuilds_correctly;
#[cfg(test)]
pub use to_pattern_set_rebuilds_correctly::*;
#[cfg(test)]
pub mod trailing_bytes_rejected;
#[cfg(test)]
pub use trailing_bytes_rejected::*;
#[cfg(test)]
pub mod oversized_packed_bytes_length_detected;
#[cfg(test)]
pub use oversized_packed_bytes_length_detected::*;
#[cfg(test)]
pub mod invalid_utf8_in_name_detected;
#[cfg(test)]
pub use invalid_utf8_in_name_detected::*;
#[cfg(test)]
pub mod negative_length_prefixed_fields;
#[cfg(test)]
pub use negative_length_prefixed_fields::*;
#[cfg(test)]
pub mod mixed_literals_and_regex_roundtrip;
#[cfg(test)]
pub use mixed_literals_and_regex_roundtrip::*;
#[cfg(test)]
pub mod single_literal_optimization_preserved;
#[cfg(test)]
pub use single_literal_optimization_preserved::*;
#[cfg(test)]
pub mod very_long_literal_pattern;
#[cfg(test)]
pub use very_long_literal_pattern::*;
#[cfg(test)]
pub mod overlapping_patterns_preserved;
#[cfg(test)]
pub use overlapping_patterns_preserved::*;
#[cfg(test)]
pub mod unicode_literals_preserved;
#[cfg(test)]
pub use unicode_literals_preserved::*;
