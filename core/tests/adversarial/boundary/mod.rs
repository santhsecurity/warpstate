use warpstate::*;

#[test]
fn test_100k_character_input() {
    let ps = PatternSet::builder().literal("TARGET").build().unwrap();
    let size = 100_000;
    let mut data = vec![b'A'; size];

    // Put target at the beginning, middle, and end
    data[0..6].copy_from_slice(b"TARGET");
    data[50_000..50_006].copy_from_slice(b"TARGET");
    data[size - 6..].copy_from_slice(b"TARGET");

    let matches = ps.scan(&data).unwrap();
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].start, 0);
    assert_eq!(matches[1].start, 50_000);
    assert_eq!(matches[2].start, (size - 6) as u32);
}
