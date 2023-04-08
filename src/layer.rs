use crate::{activation::Activation, initializer::Initializer, tensor::Tensor};

/// A single layer in a neural network.
#[derive(Debug)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub data: Tensor,
}

impl Layer {
    /// Creates a new `Layer` with the specified number of input and output neurons,
    /// using the specified initialization technique.
    pub fn new(f_in: usize, f_out: usize, initializer: &Initializer) -> Layer {
        let weights = Tensor::initialize(f_in, f_out, initializer);
        let biases = Tensor::initialize(f_out, 1, initializer);
        let data = Tensor::zeros(f_out, 1);

        Layer {
            weights,
            biases,
            data,
        }
    }

    /// Feeds the input through the layer, returning the output.
    pub fn feed_forward(&mut self, inputs: &Tensor, activation: Activation) -> Tensor {
        let output = (self.weights)
            .mul(inputs)
            .add(&self.biases)
            .mapv(&|x| activation.apply(x));

        self.data = output.clone();

        output
    }

    /// Performs backpropagation on the layer, using the specified error and learning rate.
    pub fn back_propagate(
        &mut self,
        error: &mut Tensor,
        activation: Activation,
        learning_rate: f64,
    ) -> Tensor {
        let gradients = self.data.clone().mapv(&|x| activation.gradient(x));
        let delta = error.mul(&gradients);
        let weights_delta = (self.weights)
            .transpose()
            .mul(&delta)
            .mul_scalar(learning_rate);
        let bias_delta = delta.mul_scalar(learning_rate);

        self.weights = self.weights.sub(&self.data.mul(&weights_delta.transpose()));
        self.biases = self.biases.sub(&bias_delta);

        weights_delta
    }
}
