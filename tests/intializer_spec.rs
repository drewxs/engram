use engram::initializer::Initializer;

#[test]
fn test_xavier() {
    let weights = Initializer::Xavier.initialize(2, 3);
    assert_eq!(weights.len(), 6);
    println!("{:?}", weights);
    assert!(weights.iter().all(|w| w.abs() <= 1.0));
}

#[test]
fn test_kaiming() {
    let weights = Initializer::Kaiming.initialize(4, 3);
    assert_eq!(weights.len(), 12);
    println!("{:?}", weights);
    assert!(weights.iter().all(|w| w.abs() <= 1.0));
}
