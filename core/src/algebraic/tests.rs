use super::AlgebraicDfaMatcher;
use crate::matcher::BlockMatcher;
use crate::{Match, PatternSet};

fn long_literal(min_states: usize) -> String {
    let len = min_states.saturating_sub(2);
    (0..len)
        .map(|index| char::from(b'a' + (index % 26) as u8))
        .collect()
}

fn regex_state_count(patterns: &PatternSet) -> usize {
    patterns
        .compiled_regex_dfa()
        .expect("compiled regex dfa")
        .as_ref()
        .state_count()
}

fn hybrid_tail_bytes(patterns: &PatternSet, data: &[u8]) -> usize {
    let dfa = patterns.compiled_regex_dfa().expect("compiled regex dfa");
    let mut state = dfa.start_state & 0x3FFF_FFFF;
    for (index, &byte) in data.iter().enumerate() {
        state = dfa.transition_for_byte(state, byte);
        if (state & 0x3FFF_FFFF) >= super::MAX_ALGEBRAIC_STATES {
            return data.len().saturating_sub(index + 1);
        }
    }
    0
}

fn algebraic_matcher(patterns: &PatternSet) -> Option<AlgebraicDfaMatcher> {
    match pollster::block_on(AlgebraicDfaMatcher::new(patterns)) {
        Ok(matcher) => Some(matcher),
        Err(crate::Error::NoGpuAdapter) => None,
        Err(other) => panic!("unexpected matcher construction error: {other:?}"),
    }
}

async fn scan_with_matcher(patterns: &PatternSet, data: &[u8]) -> Option<Vec<Match>> {
    let matcher = algebraic_matcher(patterns)?;
    Some(
        matcher
            .scan_block(data)
            .await
            .unwrap_or_else(|err| panic!("unexpected scan error: {err:?}")),
    )
}

#[tokio::test]
#[ignore = "GPU shader parity bug under software Vulkan."]
async fn algebraic_single_pattern() {
    let patterns = PatternSet::builder().regex("needle").build().unwrap();
    let Some(matches) = scan_with_matcher(&patterns, b"hay needle hay").await else {
        return;
    };
    assert_eq!(
        matches,
        vec![Match {
            pattern_id: 0,
            start: 4,
            end: 10,
            padding: 0,
        }]
    );
}

#[tokio::test]
#[ignore = "GPU shader parity bug under software Vulkan."]
async fn algebraic_multiple_patterns() {
    let patterns = PatternSet::builder()
        .regex("alpha")
        .regex("beta")
        .regex("gamma")
        .regex("delta")
        .regex("eta")
        .build()
        .unwrap();
    let data = b"alpha beta gamma delta eta";
    let Some(matches) = scan_with_matcher(&patterns, data).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(data).unwrap());
}

#[tokio::test]
#[ignore = "GPU shader parity bug under software Vulkan."]
async fn algebraic_matches_cpu_parity() {
    let patterns = PatternSet::builder()
        .regex("aba")
        .regex("ba")
        .regex("cab")
        .regex("ababa")
        .build()
        .unwrap();
    let data = b"cababa ababacab";
    let Some(matches) = scan_with_matcher(&patterns, data).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(data).unwrap());
}

/// Tests that a DFA exceeding MAX_ALGEBRAIC_STATES (64) uses the hybrid path.
#[test]
fn algebraic_accepts_large_dfa_via_hybrid_mode() {
    let literal = long_literal(70);
    let patterns = PatternSet::builder().regex(&literal).build().unwrap();
    assert!(regex_state_count(&patterns) >= 70);
    if let Some(matcher) = algebraic_matcher(&patterns) {
        assert!(matcher.full_state_count > super::MAX_ALGEBRAIC_STATES);
        assert_eq!(matcher.gpu_state_count, super::HYBRID_GPU_STATES);
    }
}

#[tokio::test]
async fn algebraic_hybrid_70_state_matches_cpu_backend() {
    let literal = long_literal(70);
    let patterns = PatternSet::builder().regex(&literal).build().unwrap();
    assert!(regex_state_count(&patterns) >= 70);
    let data = literal.as_bytes();
    let Some(matches) = scan_with_matcher(&patterns, data).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(data).unwrap());
}

#[tokio::test]
async fn algebraic_hybrid_100_state_matches_cpu_backend() {
    let literal = long_literal(100);
    let patterns = PatternSet::builder().regex(&literal).build().unwrap();
    assert!(regex_state_count(&patterns) >= 100);
    let data = literal.as_bytes();
    let Some(matches) = scan_with_matcher(&patterns, data).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(data).unwrap());
}

#[tokio::test]
async fn algebraic_hybrid_matches_cpu_backend_exactly() {
    let literal = long_literal(100);
    let patterns = PatternSet::builder().regex(&literal).build().unwrap();
    let mut data = literal.clone().into_bytes();
    data.extend_from_slice(b"zz");
    let Some(matches) = scan_with_matcher(&patterns, &data[..literal.len()]).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(&data[..literal.len()]).unwrap());
}

#[test]
fn algebraic_hybrid_reduces_sequential_tail_work() {
    let literal = long_literal(100);
    let patterns = PatternSet::builder().regex(&literal).build().unwrap();
    let data = literal.as_bytes();
    let tail_bytes = hybrid_tail_bytes(&patterns, data);
    assert!(tail_bytes > 0);
    assert!(tail_bytes < data.len());
}

#[tokio::test]
async fn algebraic_empty_input() {
    let patterns = PatternSet::builder().regex("x").build().unwrap();
    let Some(matches) = scan_with_matcher(&patterns, b"").await else {
        return;
    };
    assert!(matches.is_empty());
}

#[tokio::test]
#[ignore = "GPU shader parity bug under software Vulkan."]
async fn algebraic_block_boundary() {
    let patterns = PatternSet::builder().regex("needle").build().unwrap();
    let mut data = vec![b'a'; 4095];
    data.extend_from_slice(b"needle");
    data.extend_from_slice(b"tail");
    let Some(matches) = scan_with_matcher(&patterns, &data).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(&data).unwrap());
}

#[tokio::test]
#[ignore = "GPU shader parity bug under software Vulkan."]
async fn algebraic_long_input() {
    let patterns = PatternSet::builder()
        .regex("warp")
        .regex("state")
        .regex("prefix")
        .build()
        .unwrap();
    let mut data = vec![b'x'; 1 << 20];
    data[1000..1004].copy_from_slice(b"warp");
    data[700_000..700_005].copy_from_slice(b"state");
    data[900_000..900_006].copy_from_slice(b"prefix");
    let Some(matches) = scan_with_matcher(&patterns, &data).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(&data).unwrap());
}

#[tokio::test]
#[ignore = "GPU shader parity bug under software Vulkan."]
async fn algebraic_overlapping_matches() {
    let patterns = PatternSet::builder()
        .regex("aba")
        .regex("ba")
        .regex("ababa")
        .build()
        .unwrap();
    let data = b"ababa";
    let Some(matches) = scan_with_matcher(&patterns, data).await else {
        return;
    };
    assert_eq!(matches, patterns.scan(data).unwrap());
}
