//! Create and manipulate neural network layers.
//!
//! This module provides a Layer struct for representing a single layer in a neural network,
//! along with methods for feeding inputs through the layer and performing backpropagation.

use crate::{Activation, Initializer, Loss, Optimize, Optimizer, Tensor};

/// A single layer in a neural network.
#[derive(Debug)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
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
    /// # use engram::*;
    ///
    /// let layer = Layer::new(2, 3, &Initializer::Xavier, Activation::Sigmoid);
    ///
    /// assert_eq!(layer.weights.shape(), (2, 3));
    /// assert_eq!(layer.biases.shape(), (3, 1));
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
            biases: Tensor::initialize(f_out, 1, initializer),
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
    /// # use engram::*;
    ///
    /// let mut layer = Layer::new(3, 4, &Initializer::Xavier, Activation::Sigmoid);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0], [1.0, 2.0, 3.0]];
    /// let output = layer.feed_forward(&inputs);
    ///
    /// assert_eq!(output.shape(), (4, 4));
    /// ```
    pub fn feed_forward(&self, inputs: &Tensor) -> Tensor {
        let output = inputs.matmul(&self.weights);
        let output = output.add(&self.biases).activate(&self.activation);

        output
    }

    /// Feeds the input through the layer, updating the layer's inputs and output.
    /// This method is used when training the network.
    /// If you are only evaluating the network, use `feed_forward` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut layer = Layer::new(3, 4, &Initializer::Xavier, Activation::Sigmoid);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0], [1.0, 2.0, 3.0]];
    /// layer.feed_forward_mut(&inputs);
    ///
    /// assert_eq!(layer.output.unwrap().shape(), (4, 4));
    /// ```
    pub fn feed_forward_mut(&mut self, inputs: &Tensor) {
        let output = inputs.matmul(&self.weights);
        let output = output.add(&self.biases).activate(&self.activation);

        self.inputs = Some(inputs.clone());
        self.output = Some(output);
    }

    /// Performs backpropagation on the layer based on the provided targets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut layer = Layer::new(3, 2, &Initializer::Xavier, Activation::Sigmoid);
    /// let inputs = tensor![[1.0, 2.0, 7.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0], [1.0, 2.0, 3.0]];
    /// let targets = tensor![[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
    ///
    /// layer.feed_forward_mut(&inputs);
    ///
    /// let mean_loss = layer.back_propagate(&targets, &Loss::MeanSquaredError, &mut Optimizer::SGD { learning_rate: 0.1 });
    /// let predicted_loss = 0.6;
    ///
    /// assert!((mean_loss - predicted_loss).abs() < 0.1);
    /// ```
    pub fn back_propagate(
        &mut self,
        targets: &Tensor,
        loss_function: &Loss,
        optimizer: &mut Optimizer,
    ) -> f64 {
        let predictions = self
            .output
            .as_ref()
            .expect("Call to back_propagate without calling feed_forward first!");

        let loss = loss_function.loss(&predictions, &targets);
        let mut gradient = loss_function.gradient(&predictions, &targets);

        optimizer.step(&mut self.weights, &mut gradient.clone());
        optimizer.step(&mut self.biases, &mut gradient);

        loss.mean()
    }
}
