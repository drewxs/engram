use engram::initializer::Initializer::{Kaiming, Xavier};

#[test]
fn test_xavier() {
    let weights = Xavier.initialize(2, 3);
    println!("{:?}", weights);
    assert!(weights.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
}

#[test]
fn test_kaiming() {
    let weights = Kaiming.initialize(4, 3);
    println!("{:?}", weights);
    assert!(weights.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
}
