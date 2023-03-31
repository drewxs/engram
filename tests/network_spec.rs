use engram::activation::Activation;
use engram::initializer::Initializer;
use engram::network::Network;
use engram::tensor;

#[test]
fn test_network() {
    // Input data/labels for training XOR
    let layers = vec![2, 3, 4];
    let inputs = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = tensor![[0.0], [1.0], [1.0], [0.0]];

    let mut network = Network::new(layers, Initializer::Xavier, Activation::Sigmoid, 0.5);

    network.train(inputs, targets, 10000);

    let tolerance = 0.2;
    let y_hat = network.feed_forward(vec![0.0, 0.0])[0];

    println!("y_hat: {:?}", y_hat);
    println!("y: {:?}", 0.0);

    assert!((y_hat - 0.0).abs() < tolerance);
}
