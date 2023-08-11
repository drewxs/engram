//! A module for creating and manipulating neural network layers.
//!
//! This module provides a Layer struct for representing a single layer in a neural network,
//! along with methods for feeding inputs through the layer and performing backpropagation.

use crate::{Activation, Initializer, Tensor};

/// A single layer in a neural network.
#[derive(Debug)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub d_weights: Tensor,
    pub d_biases: Tensor,
    pub output: Option<Tensor>,
}

impl Layer {
    /// Creates a new `Layer` with the specified number of input and output neurons,
    /// using the specified initialization technique.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Layer, Initializer};
    /// let layer = Layer::new(2, 3, &Initializer::Xavier);
    /// assert_eq!(layer.weights.shape(), (3, 2));
    /// assert_eq!(layer.biases.shape(), (3, 1));
    /// assert_eq!(layer.d_weights.shape(), (3, 2));
    /// assert_eq!(layer.d_biases.shape(), (3, 1));
    /// assert!(layer.output.is_none());
    /// ```
    pub fn new(f_in: usize, f_out: usize, initializer: &Initializer) -> Layer {
        let weights = Tensor::initialize(f_out, f_in, initializer);
        let biases = Tensor::initialize(f_out, 1, initializer);
        let d_weights = Tensor::zeros(f_out, f_in);
        let d_biases = Tensor::zeros(f_out, 1);
        let output = None;

        Layer {
            weights,
            biases,
            d_weights,
            d_biases,
            output,
        }
    }

    /// Feeds the input through the layer, returning the output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Activation, Initializer, Layer, tensor};
    /// let mut layer = Layer::new(2, 3, &Initializer::Xavier);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0]];
    /// let output = layer.feed_forward(&inputs, Activation::Sigmoid);
    /// assert_eq!(output.shape(), (3, 2));
    /// ```
    pub fn feed_forward(&mut self, inputs: &Tensor, activation: Activation) -> Tensor {
        let weighted_sum = inputs.matmul(&self.weights);
        let biases = self.biases.broadcast_to(&weighted_sum);
        let output = activation.apply_tensor(&weighted_sum.add(&biases));

        self.output = Some(output.clone());
        output
    }

    /// Performs backpropagation on the layer based on the provided targets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Activation, Initializer, Layer, tensor};
    /// let mut layer = Layer::new(2, 3, &Initializer::Xavier);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0]];
    /// layer.feed_forward(&inputs, Activation::Sigmoid);
    /// let targets = tensor![[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]];
    /// let error = layer.back_propagate(&targets, Activation::Sigmoid);
    /// assert_eq!(error.shape(), (3, 3));
    /// ```
    pub fn back_propagate(&mut self, targets: &Tensor, activation: Activation) -> Tensor {
        let output = match &self.output {
            Some(output) => output,
            None => panic!("Call to back_propagate without calling feed_forward first!"),
        };

        // Compute error delta for this layer
        let error = output.sub(&targets);
        let d_output = activation.gradient_tensor(&output);
        let error_delta = error.mul(&d_output);

        // Compute gradients for weights and biases
        // Update weights and biases gradients
        self.d_weights = self.weights.transpose().matmul(&error_delta);
        self.d_biases = error_delta.clone();

        // Compute error for previous layer
        error_delta.matmul(&self.weights.transpose())
    }
}
