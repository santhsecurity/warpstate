//! Adversarial tests for compiled index serialization.
//!
//! warpscan scans the ENTIRE internet's supply chain. A corrupted index means malware goes undetected.
//! Every finding is critical at internet scale.

use super::{CompiledPatternIndex, VERSION};

// =============================================================================
// TEST 1: Wrong magic bytes — must error, not panic
// =============================================================================

pub mod empty_pattern_set_errors_on_build;
#[cfg(test)]
pub mod invalid_utf8_in_name_detected;
pub mod large_pattern_set_100k_does_not_oom;
pub mod load_future_version_errors;
pub mod load_literal_count_mismatch_detected;
#[cfg(test)]
pub mod load_literal_count_zero_with_offsets;
#[cfg(test)]
pub mod load_max_version_errors;
#[cfg(test)]
pub mod load_offset_overflow_detected;
pub mod load_offset_past_packed_bytes_detected;
pub mod load_truncated_at_various_boundaries;
#[cfg(test)]
pub mod load_truncated_header_fields;
#[cfg(test)]
pub mod load_version_zero_errors;
#[cfg(test)]
pub mod load_wrong_magic_bytes_errors;
#[cfg(test)]
pub mod medium_pattern_set_10k_roundtrip;
#[cfg(test)]
pub mod mixed_literals_and_regex_roundtrip;
#[cfg(test)]
pub mod multiple_patterns_with_null_bytes;
#[cfg(test)]
pub mod names_count_mismatch_detected;
#[cfg(test)]
pub mod negative_length_prefixed_fields;
pub mod null_bytes_pattern_roundtrip;
#[cfg(test)]
pub mod overlapping_patterns_preserved;
#[cfg(test)]
pub mod oversized_packed_bytes_length_detected;
pub mod pattern_count_mismatch_detected;
#[cfg(test)]
pub mod pattern_ending_with_null;
#[cfg(test)]
pub mod pattern_starting_with_null;
#[cfg(test)]
pub mod proptest_roundtrip_preserves_scan_results;
#[cfg(test)]
pub mod round_trip_case_insensitive_results_identical;
pub mod round_trip_literal_and_regex_results_identical;
#[cfg(test)]
pub mod single_literal_optimization_preserved;
#[cfg(test)]
pub mod to_pattern_set_rebuilds_correctly;
#[cfg(test)]
pub mod trailing_bytes_rejected;
#[cfg(test)]
pub mod unicode_literals_preserved;
#[cfg(test)]
pub mod very_long_literal_pattern;
