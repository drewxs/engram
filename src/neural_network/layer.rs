//! Create and manipulate neural network layers.
//!
//! This module provides a Layer struct for representing a single layer in a neural network,
//! along with methods for feeding inputs through the layer and performing backpropagation.

use crate::{Activation, Initializer, Optimizer, Tensor};

/// A single layer in a neural network.
#[derive(Debug)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub d_weights: Tensor,
    pub d_biases: Tensor,
    pub inputs: Option<Tensor>,
    pub output: Option<Tensor>,
    pub activation: Activation,
}

impl Layer {
    /// Creates a new `Layer` with the specified number of input and output neurons,
    /// using the specified initialization technique.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Activation, Initializer, Layer};
    /// let layer = Layer::new(2, 3, &Initializer::Xavier, Activation::Sigmoid);
    /// assert_eq!(layer.weights.shape(), (2, 3));
    /// assert_eq!(layer.biases.shape(), (2, 1));
    /// assert_eq!(layer.d_weights.shape(), (2, 3));
    /// assert_eq!(layer.d_biases.shape(), (2, 1));
    /// assert!(layer.output.is_none());
    /// ```
    pub fn new(
        f_in: usize,
        f_out: usize,
        initializer: &Initializer,
        activation: Activation,
    ) -> Layer {
        Layer {
            weights: Tensor::initialize(f_in, f_out, initializer),
            biases: Tensor::initialize(f_in, 1, initializer),
            d_weights: Tensor::zeros(f_in, f_out),
            d_biases: Tensor::zeros(f_in, 1),
            inputs: None,
            output: None,
            activation,
        }
    }

    /// Feeds the input through the layer, returning the output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Activation, Initializer, Layer, tensor};
    /// let mut layer = Layer::new(3, 2, &Initializer::Xavier, Activation::Sigmoid);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0], [1.0, 2.0, 3.0]];
    /// let output = layer.feed_forward(&inputs);
    /// assert_eq!(output.shape(), (4, 2));
    /// ```
    pub fn feed_forward(&mut self, inputs: &Tensor) -> Tensor {
        let weighted_sum = inputs.matmul(&self.weights);
        let biases = self.biases.broadcast_to(&weighted_sum);
        let output = weighted_sum.add(&biases).activate(&self.activation);

        self.inputs = Some(inputs.clone());
        self.output = Some(output.clone());
        output
    }

    /// Performs backpropagation on the layer based on the provided targets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Activation, Initializer, Layer, tensor};
    /// let mut layer = Layer::new(3, 2, &Initializer::Xavier, Activation::Sigmoid);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0], [1.0, 2.0, 3.0]];
    /// let output = layer.feed_forward(&inputs);
    /// let targets = tensor![[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]];
    /// layer.back_propagate(&targets);
    /// ```
    pub fn back_propagate(&mut self, targets: &Tensor) {
        let output = match &self.output {
            Some(output) => output,
            None => panic!("Call to back_propagate without calling feed_forward first!"),
        };

        // Compute error delta for this layer
        let error = output.sub(&targets);
        let d_output = output.activate(&self.activation);
        let error_delta = error.mul(&d_output);

        let inputs = self.inputs.as_ref().unwrap();
        let num_samples = inputs.rows as f64;

        // Compute gradients for weights and biases
        self.d_weights = inputs
            .transpose()
            .matmul(&error_delta)
            .div_scalar(num_samples);
        self.d_biases = error_delta
            .sum_axis(1)
            .div_scalar(num_samples)
            .resize_to(&self.biases);

        // Update weights and biases based on gradients
        self.weights.sub_assign(&self.d_weights);
        self.biases.sub_assign(&self.d_biases);
    }

    /// Performs a single step of gradient descent on the layer's weights and biases.
    ///
    /// # Examples
    ///
    /// # use engram::{Activation, Initializer, Layer, Optimizer, tensor};
    /// let mut layer = Layer::new(2, 3, &Initializer::Xavier, Activation::Sigmoid);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0]];
    /// layer.feed_forward(&inputs);
    /// let targets = tensor![[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]];
    /// layer.back_propagate(&targets);
    /// layer.step(&mut Optimizer::SGD { learning_rate: 0.1 });
    pub fn step(&mut self, optimizer: &mut Optimizer) {
        optimizer.step(&mut self.weights, &mut self.d_weights);
        optimizer.step(&mut self.biases, &mut self.d_biases);
    }
}