use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::{Anchored, Input};

fn main() {
    let dfa = dense::Builder::new()
        .configure(dense::Config::new().start_kind(StartKind::Both))
        .syntax(regex_automata::util::syntax::Config::new().multi_line(true))
        .build_many(&["^fn main$", "mod"])
        .unwrap();

    let input = b"mod demo;\nfn main\nfn main()\n";
    let mut pos = 0;
    while pos < input.len() {
        if let Some(hm) = dfa
            .try_search_fwd(&Input::new(input).range(pos..).anchored(Anchored::No))
            .unwrap()
        {
            println!("Match start..end: {}..{}", pos, hm.offset());
            println!("Match pat: {}", hm.pattern().as_usize());
            pos = hm.offset().max(pos + 1);
        } else {
            break;
        }
    }
}
