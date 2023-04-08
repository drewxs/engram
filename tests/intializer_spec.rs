use engram::Initializer;

#[test]
fn test_xavier() {
    let weights = Initializer::Xavier.initialize(2, 3);
    println!("{:?}", weights);
    assert!(weights.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
}

#[test]
fn test_kaiming() {
    let weights = Initializer::Kaiming.initialize(4, 3);
    println!("{:?}", weights);
    assert!(weights.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
}
