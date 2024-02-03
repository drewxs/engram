use crate::{linalg::Tensor, Activation, Initializer};

#[derive(Debug, Clone)]
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activation,
}

impl Layer {
    pub fn new(
        f_in: usize,
        f_out: usize,
        initializer: Initializer,
        activation: Activation,
    ) -> Layer {
        Layer {
            inputs: f_in,
            outputs: f_out,
            weights: Tensor::initialize(f_out, f_in, &initializer),
            biases: Tensor::initialize(f_out, 1, &initializer),
            activation,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.weights
            .matmul(&input.transpose())
            .transpose()
            .add(&self.biases)
            .activate(&self.activation)
    }

    pub fn backward(&self, input: &Tensor, d_output: &Tensor) -> (Tensor, Tensor, Tensor) {
        let d_weight = input.transpose().matmul(d_output);
        let d_bias = d_output.sum_axis(0);
        let mut d_input = if d_output.is_matmul_compatible(&self.weights) {
            d_output.matmul(&self.weights)
        } else {
            d_output.transpose().matmul(&self.weights.transpose())
        };
        if self.outputs != 1 {
            d_input.transpose_mut()
        }

        (d_input, d_weight, d_bias)
    }
}
