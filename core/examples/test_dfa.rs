use regex_automata::{dfa::dense, dfa::Automaton, Input};

fn main() {
    let patterns = vec!["password", "secret_key"];
    let dfa = dense::Builder::new().build_many(&patterns).unwrap();

    // special_max, dead_max etc don't exist directly on `DFA`, they are on `StateID` classes or accessed differently?
    // Actually, state class boundaries are typically accessed via dfa.special_max() in some versions, or maybe not public.
    // Let's print something easily accessible:
    println!(
        "Start state: {:?}",
        dfa.start_state_forward(&Input::new("")).unwrap()
    );

    // We can do a BFS to build our own simple table!
    let mut states = vec![dfa.start_state_forward(&Input::new("")).unwrap()];
    let mut i = 0;
    while i < states.len() {
        let state = states[i];
        for b in 0..=255 {
            let next = dfa.next_state(state, b);
            if !states.contains(&next) {
                states.push(next);
            }
        }
        i += 1;
    }
    println!("Total states discovered via BFS: {}", states.len());
}
