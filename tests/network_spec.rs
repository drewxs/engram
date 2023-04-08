use engram::{tensor, Activation, Initializer, Network, Optimizer};

#[test]
fn test_network() {
    // Input data/labels for training XOR
    let layers = vec![2, 3, 4];
    let inputs = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = tensor![[0.0], [1.0], [1.0], [0.0]];

    let mut network = Network::new(
        layers,
        Initializer::Xavier,
        Activation::Sigmoid,
        Optimizer::SGD { learning_rate: 0.1 },
        0.5,
    );

    network.train(&inputs, &targets, 10, 100);

    let tolerance = 0.2;
    let y_hat = network.feed_forward(&tensor![[0.0, 0.0]]).first();

    println!("y_hat: {:?}", y_hat);
    println!("y: {:?}", 0.0);

    assert!((y_hat - 0.0).abs() < tolerance);
}
