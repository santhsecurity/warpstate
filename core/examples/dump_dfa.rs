fn main() {
    let patterns = vec!["he", "hello", "world"];
    let ids = vec![0, 1, 2];
    let dfa = warpstate::dfa::RegexDFA::build(&patterns, &ids).unwrap();

    // Dump state progression for "hello world"
    let data = b"hello world";
    let mut state = dfa.start_state();

    let class_count = dfa.class_count() as usize;

    for (i, &b) in data.iter().enumerate() {
        let state_idx = (state & 0x3FFF_FFFF) as usize;
        let class_id = dfa.byte_classes()[b as usize] as usize;
        state = dfa.transition_table()[state_idx * class_count + class_id];

        let is_match = (state & 0x8000_0000) != 0;
        let is_dead = (state & 0x4000_0000) != 0;
        println!(
            "i={}, byte='{}', state_idx={}, is_match={}, is_dead={}",
            i,
            b as char,
            state & 0x3FFF_FFFF,
            is_match,
            is_dead
        );
        if is_match {
            let ptr = dfa.match_list_pointers()[(state & 0x3FFF_FFFF) as usize];
            let qty = dfa.match_lists()[ptr as usize];
            print!("  Matches: ");
            for m in 0..qty {
                let pat_id = dfa.match_lists()[(ptr + 1 + m) as usize];
                print!("pat{} ", pat_id);
            }
            println!("");
        }
    }
}
