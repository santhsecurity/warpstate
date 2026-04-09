//! Property-based parity tests for warpstate.
//!
//! Verifies invariants across backends, serialization, and edge cases.

use proptest::prelude::*;
use warpstate::compiled_index::CompiledPatternIndex;
use warpstate::{cpu, PatternSet};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10000))]

    /// 1. CPU scan == PatternSet::scan for all inputs (CPU backend parity)
    #[test]
    fn prop_cpu_scan_parity(
        input in prop::collection::vec(any::<u8>(), 0..5000),
        patterns_data in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..32), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let ps = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let res1 = ps.scan(&input).unwrap();
        let mut out_matches = [warpstate::Match::from_parts(0, 0, 0); 1000];
        let count = cpu::scan(ps.ir(), &input, &mut out_matches).unwrap();
        let res2 = out_matches[..count].to_vec();

        prop_assert_eq!(res1.len(), res2.len());
        prop_assert_eq!(&res1[..], &res2[..]);
    }

    /// 2. scan(empty_input) == [] for all pattern sets
    #[test]
    fn prop_scan_empty_input_is_empty(
        patterns_data in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..32), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let ps = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = ps.scan(b"").unwrap();
        prop_assert!(matches.is_empty());
    }

    /// 3. scan results are sorted by (start, pattern_id)
    #[test]
    fn prop_scan_results_sorted(
        input in prop::collection::vec(any::<u8>(), 0..5000),
        patterns_data in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..16), 1..10)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let ps = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = ps.scan(&input).unwrap();

        for window in matches.windows(2) {
            let m1 = &window[0];
            let m2 = &window[1];
            // Primary sort: start, secondary sort: pattern_id, tertiary: end
            prop_assert!(
                (m1.start, m1.pattern_id, m1.end) <= (m2.start, m2.pattern_id, m2.end),
                "Matches not sorted by (start, pattern_id, end): {:?} vs {:?}", m1, m2
            );
        }
    }

    /// 4. All match.start < match.end
    /// 5. All match.end <= input.len()
    #[test]
    fn prop_match_bounds_valid(
        input in prop::collection::vec(any::<u8>(), 0..5000),
        patterns_data in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..32), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let ps = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = ps.scan(&input).unwrap();
        let input_len = input.len() as u32;

        for m in &matches {
            prop_assert!(m.start < m.end, "Match start {} must be < end {}", m.start, m.end);
            prop_assert!(m.end <= input_len, "Match end {} must be <= input len {}", m.end, input_len);
        }
    }

    /// 6. PatternSet::builder().literal(x).build().scan(x) always finds at least 1 match
    #[test]
    fn prop_literal_matches_itself(
        x in prop::collection::vec(any::<u8>(), 1..100)
    ) {
        let ps = match PatternSet::builder().literal_bytes(x.clone()).build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let matches = ps.scan(&x).unwrap();
        prop_assert!(!matches.is_empty(), "Pattern {:?} should match itself", x);

        // Ensure it matches at least once at start 0
        prop_assert!(matches.iter().any(|m| m.start == 0), "Pattern should match at least at offset 0");
    }

    /// 7. parse(serialize(PatternIR)) == original (via CompiledPatternIndex behavior)
    #[test]
    fn prop_compiled_index_parity(
        input in prop::collection::vec(any::<u8>(), 0..5000),
        patterns_data in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..32), 1..5)
    ) {
        let mut builder = PatternSet::builder();
        for p in &patterns_data {
            builder = builder.literal_bytes(p.clone());
        }
        let ps = match builder.build() {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        let serialized = CompiledPatternIndex::build(&ps).unwrap();
        let index = CompiledPatternIndex::load(&serialized).unwrap();

        let ps_matches = ps.scan(&input).unwrap();
        let index_matches = index.scan(&input).unwrap();

        prop_assert_eq!(ps_matches, index_matches, "Serialized index scan results differ from original PatternSet");

        // Also check key metadata roundtrip
        let ir = ps.ir();
        let literals = index.parse_literals().unwrap();
        prop_assert_eq!(&ir.packed_bytes, &literals.packed_bytes);
        prop_assert_eq!(&ir.offsets, &literals.offsets);

        let index_dfas = index.parse_regex_dfas().unwrap();
        let ps_dfas = ir.regex_dfas();
        prop_assert_eq!(ps_dfas.len(), index_dfas.len());
        for (p, i) in ps_dfas.iter().zip(index_dfas.iter()) {
            prop_assert_eq!(p.transition_table(), i.transition_table());
            prop_assert_eq!(p.match_list_pointers(), i.match_list_pointers());
            prop_assert_eq!(p.match_lists(), i.match_lists());
            prop_assert_eq!(p.pattern_lengths(), i.pattern_lengths());
            prop_assert_eq!(p.start_state(), i.start_state());
            prop_assert_eq!(p.class_count(), i.class_count());
            prop_assert_eq!(p.eoi_class(), i.eoi_class());
            prop_assert_eq!(p.byte_classes(), i.byte_classes());
        }
    }
}
