use regex_automata::{
    dfa::{dense, Automaton},
    Anchored, Input,
};

fn main() {
    let pattern = "a.*b";
    let compiled = dense::Builder::new()
        .configure(
            dense::Config::new()
                .match_kind(regex_automata::MatchKind::All)
                .start_kind(regex_automata::dfa::StartKind::Both),
        )
        .syntax(
            regex_automata::util::syntax::Config::new()
                .multi_line(true)
                .unicode(true),
        )
        .build(pattern)
        .unwrap();

    let haystack = b"a b b";
    let mut pos = 0;
    while pos < haystack.len() {
        let input = Input::new(haystack).range(pos..).anchored(Anchored::No);
        if let Ok(Some(m)) = compiled.try_search_fwd(&input) {
            println!("match at pos {}: offset {}", pos, m.offset());

            // To find overlapping starts, we do a backwards search
            let match_start = m.offset();
            let mut best_start = match_start;
            for candidate in (0..m.offset()).rev() {
                let input_a = Input::new(haystack)
                    .range(candidate..)
                    .anchored(Anchored::Yes);
                if let Ok(Some(hm)) = compiled.try_search_fwd(&input_a) {
                    if hm.offset() == m.offset() {
                        best_start = candidate;
                        println!("found start at {}", best_start);
                    }
                }
            }

            // Now, where do we resume pos? If we do best_start + 1, we should find another end... right?
            pos = best_start + 1;
        } else {
            break;
        }
    }
}
