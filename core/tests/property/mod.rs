use proptest::collection::vec;
use proptest::prelude::*;
use warpstate::*;

fn gen_input() -> impl Strategy<Value = Vec<u8>> {
    vec(any::<u8>(), 0..10000)
}

fn gen_patterns() -> impl Strategy<Value = Vec<String>> {
    vec(
        proptest::string::string_regex("[a-z]{2,20}").unwrap(),
        1..10,
    )
}

fn gen_patterns_pair() -> impl Strategy<Value = (Vec<String>, Vec<String>)> {
    (gen_patterns(), gen_patterns())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]


    #[test]
    fn prop_scan_literal_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {
            ps.scan(input).unwrap()
        };
    }

    #[test]
    fn prop_scan_literal_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan(input).unwrap()
        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_literal_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan(input).unwrap()
        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_literal_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            ps.scan(input).unwrap()
        };
        let res2 = {
            ps.scan(input).unwrap()
        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_literal_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.literal(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.literal(p); }
        for p in &patterns2 { builder2 = builder2.literal(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;
            ps.scan(input).unwrap()
        };

        let res2 = {
            let ps = &ps2;
            ps.scan(input).unwrap()
        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_to_buffer_literal_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
    }

    #[test]
    fn prop_scan_to_buffer_literal_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_to_buffer_literal_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_to_buffer_literal_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        let res2 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_to_buffer_literal_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.literal(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.literal(p); }
        for p in &patterns2 { builder2 = builder2.literal(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        let res2 = {
            let ps = &ps2;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_with_literal_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
    }

    #[test]
    fn prop_scan_with_literal_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_with_literal_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_with_literal_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        let res2 = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_with_literal_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.literal(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.literal(p); }
        for p in &patterns2 { builder2 = builder2.literal(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };

        let res2 = {
            let ps = &ps2;

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_count_literal_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {
            ps.scan_count(input).unwrap()
        };
    }

    #[test]
    fn prop_scan_count_literal_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            ps.scan_count(input).unwrap()
        };
        let res2 = {
            ps.scan_count(input).unwrap()
        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_count_literal_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.literal(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.literal(p); }
        for p in &patterns2 { builder2 = builder2.literal(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;
            ps.scan_count(input).unwrap()
        };

        let res2 = {
            let ps = &ps2;
            ps.scan_count(input).unwrap()
        };

        assert!(res2 >= res1, "Count decreased! res1: {}, res2: {}", res1, res2);
    }

    #[test]
    fn prop_scan_overlapping_literal_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {
            ps.scan_overlapping(input).unwrap()
        };
    }

    #[test]
    fn prop_scan_overlapping_literal_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan_overlapping(input).unwrap()
        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_overlapping_literal_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan_overlapping(input).unwrap()
        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_overlapping_literal_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            ps.scan_overlapping(input).unwrap()
        };
        let res2 = {
            ps.scan_overlapping(input).unwrap()
        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_overlapping_literal_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.literal(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.literal(p); }
        for p in &patterns2 { builder2 = builder2.literal(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;
            ps.scan_overlapping(input).unwrap()
        };

        let res2 = {
            let ps = &ps2;
            ps.scan_overlapping(input).unwrap()
        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_literal_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_literal_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_literal_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_literal_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.literal(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        let res2 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_literal_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.literal(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.literal(p); }
        for p in &patterns2 { builder2 = builder2.literal(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        let res2 = {
            let ps = &ps2;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_regex_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {
            ps.scan(input).unwrap()
        };
    }

    #[test]
    fn prop_scan_regex_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan(input).unwrap()
        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_regex_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan(input).unwrap()
        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_regex_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            ps.scan(input).unwrap()
        };
        let res2 = {
            ps.scan(input).unwrap()
        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_regex_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.regex(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.regex(p); }
        for p in &patterns2 { builder2 = builder2.regex(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;
            ps.scan(input).unwrap()
        };

        let res2 = {
            let ps = &ps2;
            ps.scan(input).unwrap()
        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_to_buffer_regex_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
    }

    #[test]
    fn prop_scan_to_buffer_regex_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_to_buffer_regex_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_to_buffer_regex_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        let res2 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_to_buffer_regex_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.regex(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.regex(p); }
        for p in &patterns2 { builder2 = builder2.regex(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        let res2 = {
            let ps = &ps2;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() + 10];
            let count = ps.scan_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_with_regex_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
    }

    #[test]
    fn prop_scan_with_regex_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_with_regex_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_with_regex_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        let res2 = {

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_with_regex_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.regex(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.regex(p); }
        for p in &patterns2 { builder2 = builder2.regex(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };

        let res2 = {
            let ps = &ps2;

            let mut matches = Vec::new();
            ps.scan_with(input, |m| { matches.push(m); true }).unwrap();
            matches

        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_count_regex_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {
            ps.scan_count(input).unwrap()
        };
    }

    #[test]
    fn prop_scan_count_regex_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            ps.scan_count(input).unwrap()
        };
        let res2 = {
            ps.scan_count(input).unwrap()
        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_count_regex_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.regex(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.regex(p); }
        for p in &patterns2 { builder2 = builder2.regex(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;
            ps.scan_count(input).unwrap()
        };

        let res2 = {
            let ps = &ps2;
            ps.scan_count(input).unwrap()
        };

        assert!(res2 >= res1, "Count decreased! res1: {}, res2: {}", res1, res2);
    }

    #[test]
    fn prop_scan_overlapping_regex_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {
            ps.scan_overlapping(input).unwrap()
        };
    }

    #[test]
    fn prop_scan_overlapping_regex_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan_overlapping(input).unwrap()
        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_overlapping_regex_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {
            ps.scan_overlapping(input).unwrap()
        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_overlapping_regex_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            ps.scan_overlapping(input).unwrap()
        };
        let res2 = {
            ps.scan_overlapping(input).unwrap()
        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_overlapping_regex_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.regex(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.regex(p); }
        for p in &patterns2 { builder2 = builder2.regex(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;
            ps.scan_overlapping(input).unwrap()
        };

        let res2 = {
            let ps = &ps2;
            ps.scan_overlapping(input).unwrap()
        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_regex_no_panic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let _ = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_regex_bounds(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!(m.start < m.end, "start >= end");
            assert!(m.end as usize <= input.len(), "end > input.len");
        }
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_regex_pattern_id_valid(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let matches = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        for m in matches {
            assert!((m.pattern_id as usize) < patterns.len(), "invalid pattern id");
        }
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_regex_deterministic(input in gen_input(), patterns in gen_patterns()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns { builder = builder.regex(p); }
        let ps = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        let res2 = {

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };
        assert_eq!(res1, res2);
    }

    #[test]
    fn prop_scan_overlapping_to_buffer_regex_monotonicity(input in gen_input(), (patterns1, patterns2) in gen_patterns_pair()) {
        let input = &input;

        let mut builder = PatternSet::builder();
        for p in &patterns1 { builder = builder.regex(p); }
        let ps1 = match builder.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let mut builder2 = PatternSet::builder();
        for p in &patterns1 { builder2 = builder2.regex(p); }
        for p in &patterns2 { builder2 = builder2.regex(p); }
        let ps2 = match builder2.build() { Ok(p) => p, Err(_) => return Ok(()) };

        let res1 = {
            let ps = &ps1;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        let res2 = {
            let ps = &ps2;

            let mut matches = vec![Match::from_parts(0, 0, 0); input.len() * 10 + 10];
            let count = ps.scan_overlapping_to_buffer(input, &mut matches).unwrap();
            matches.truncate(count);
            matches

        };

        for m1 in res1 {
            assert!(res2.contains(&m1), "Match {:?} from ps1 was removed in ps2! ps1: {:?}, ps2: {:?}", m1, patterns1, patterns2);
        }
    }
}
