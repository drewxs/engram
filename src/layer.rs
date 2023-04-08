//! Implements a single layer in a neural network.
//!
//! Example:
//!
//! ```
//! use engram::{Layer, Activation, Initializer, Tensor};
//!
//! let mut layer = Layer::new(3, 2, &Initializer::Xavier);
//!
//! let inputs = tensor![[1.0], [2.0], [3.0]];
//! let output = layer.feed_forward(&inputs, Activation::Sigmoid);
//!
//! let mut error = tensor![[0.1], [-0.2]];
//! layer.back_propagate(&mut error, Activation::Sigmoid, 0.1);
//! ```

use crate::{activation::Activation, initializer::Initializer, tensor::Tensor};

/// A single layer in a neural network.
#[derive(Debug)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub weights_gradients: Tensor,
    pub biases_gradients: Tensor,
    pub output: Option<Tensor>,
}

impl Layer {
    /// Creates a new `Layer` with the specified number of input and output neurons,
    /// using the specified initialization technique.
    pub fn new(f_in: usize, f_out: usize, initializer: &Initializer) -> Layer {
        let weights = Tensor::initialize(f_in, f_out, initializer);
        let biases = Tensor::initialize(f_out, 1, initializer);
        let weights_gradients = Tensor::zeros(f_in, f_out);
        let biases_gradients = Tensor::zeros(1, f_out);
        let output = None; // should have dimensions (f_out, 1)

        Layer {
            weights,
            biases,
            weights_gradients,
            biases_gradients,
            output,
        }
    }

    /// Feeds the input through the layer, returning the output.
    pub fn feed_forward(&mut self, inputs: &Tensor, activation: Activation) -> Tensor {
        let output = (self.weights)
            .mul(inputs)
            .add(&self.biases)
            .mapv(&|x| activation.apply(x));
        self.output = Some(output.clone());
        output
    }

    /// Performs backpropagation on the layer, using the specified error and learning rate.
    pub fn back_propagate(&mut self, error: &Tensor, activation: Activation) -> Tensor {
        let output = match &self.output {
            Some(output) => output,
            None => panic!("Call to back_propagate without calling feed_forward first!"),
        };

        // Compute error delta for this layer
        let derivative = activation.gradient_tensor(&output);
        let delta = error.mul(&derivative);

        // Compute gradients for weights and biases
        // Update weights and biases gradients
        self.weights_gradients = self.weights.transpose().matmul(&delta);
        self.biases_gradients = delta.clone();

        // Compute error for previous layer
        delta.matmul(&self.weights.transpose())
    }
}
