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
            println!("match at: {}", m.offset());
            pos = m.offset() + 1; // Try to advance
        } else {
            break;
        }
    }
}
