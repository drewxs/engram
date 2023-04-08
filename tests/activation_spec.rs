use engram::Activation;

#[test]
fn it_calculates_sigmoid() {
    let activation = Activation::Sigmoid;
    let result = activation.apply(0.0);
    assert_eq!(result, 0.5);
}

#[test]
fn it_calculates_tanh() {
    let activation = Activation::TanH;
    let result = activation.apply(0.0);
    assert_eq!(result, 0.0);
}

#[test]
fn it_calculates_relu_0() {
    let activation = Activation::ReLU;
    let result = activation.apply(1.0);
    assert_eq!(result, 1.0);
}

#[test]
fn it_calculates_leaky_relu() {
    let activation = Activation::LeakyReLU;
    let result = activation.apply(-1.0);
    assert_eq!(result, -0.01);
}
