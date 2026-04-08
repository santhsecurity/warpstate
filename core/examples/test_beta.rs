use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::{Anchored, Input};

fn main() {
    let dfa = dense::Builder::new()
        .configure(dense::Config::new().start_kind(StartKind::Both))
        .build_many(&["b[0-9]+", "beta"])
        .unwrap();

    let input = b"beta";
    let hm = dfa
        .try_search_fwd(&Input::new(input).anchored(Anchored::No))
        .unwrap()
        .unwrap();
    println!("Match end: {}", hm.offset());
    println!("Match pat: {}", hm.pattern().as_usize());
}
