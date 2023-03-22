use engram::activation;

#[test]
fn it_calculates_sigmoid() {
    let result = activation::sigmoid(0.0);
    assert_eq!(result, 0.5);
}

#[test]
fn it_calculates_tanh() {
    let result = activation::tanh(0.0);
    assert_eq!(result, 0.0);
}

#[test]
fn it_calculates_relu_0() {
    let result = activation::relu(1.0);
    assert_eq!(result, 1.0);
}

#[test]
fn it_calculates_leaky_relu() {
    let result = activation::leaky_relu(-1.0);
    assert_eq!(result, -0.01);
}
