//! Multi-layer perceptron (MLP) neural network.
//!
//! A generic feed forward neural network (FNN), also known as a multi-layer perceptron (MLP).
//! Typically used for classification and regression tasks.

use crate::{Activation, Initializer, Layer, LossFunction, Optimizer, Tensor};

#[derive(Debug)]
pub struct Network {
    /// The layers in the network.
    pub layers: Vec<Layer>,
    /// The loss function used to train the network.
    pub loss_function: LossFunction,
    /// The initializer used to initialize the weights and biases in the network.
    pub initializer: Initializer,
    /// The optimizer used to optimize the weights and biases in the network.
    pub optimizer: Optimizer,
}

impl Network {
    /// Creates a new `Network` with the specified layers, activation function, learning rate, and optimizer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Network::new(
    ///     &[6, 4, 1],
    ///     Initializer::Xavier,
    ///     Activation::ReLU,
    ///     LossFunction::MeanSquaredError,
    ///     Optimizer::Adagrad {
    ///         learning_rate: 0.1,
    ///         epsilon: 1e-8,
    ///         weight_decay: Some(0.01),
    ///         shape: (3, 1),
    ///     },
    /// );
    /// assert_eq!(network.layers.len(), 2);
    /// assert_eq!(network.layers[0].weights.shape(), (6, 4));
    /// assert_eq!(network.layers[1].weights.shape(), (4, 1));
    /// assert_eq!(network.layers[0].activation, Activation::ReLU);
    /// assert_eq!(network.loss_function, LossFunction::MeanSquaredError);
    /// assert_eq!(network.optimizer, Optimizer::Adagrad {
    ///    learning_rate: 0.1,
    ///    epsilon: 1e-8,
    ///    weight_decay: Some(0.01),
    ///    shape: (3, 1),
    /// })
    /// ```
    pub fn new(
        layer_sizes: &[usize],
        initializer: Initializer,
        activation: Activation,
        loss_function: LossFunction,
        optimizer: Optimizer,
    ) -> Network {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let layer = Layer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                &initializer,
                activation.clone(),
            );
            layers.push(layer);
        }

        Network {
            layers,
            loss_function,
            initializer,
            optimizer,
        }
    }

    /// Creates a new `Network` with defaults: xavier initialization, sigmoid activation,
    /// and stochastic gradient descent optimizer with a learning rate of 0.1.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Network::default(&[6, 4, 2, 3]);
    /// assert_eq!(network.layers.len(), 3);
    /// assert_eq!(network.layers[0].weights.shape(), (6, 4));
    /// assert_eq!(network.layers[1].weights.shape(), (4, 2));
    /// assert_eq!(network.layers[2].weights.shape(), (2, 3));
    /// assert_eq!(network.layers[0].activation, Activation::Sigmoid);
    /// assert_eq!(network.loss_function, LossFunction::BinaryCrossEntropy);
    /// assert_eq!(network.optimizer, Optimizer::SGD { learning_rate: 0.1 })
    /// ```
    pub fn default(layer_sizes: &[usize]) -> Network {
        Network::new(
            layer_sizes,
            Initializer::Xavier,
            Activation::Sigmoid,
            LossFunction::BinaryCrossEntropy,
            Optimizer::SGD { learning_rate: 0.1 },
        )
    }

    /// Adds a layer to the network.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut network = Network::default(&[6, 4, 2, 3]);
    /// assert_eq!(network.layers.len(), 3);
    /// network.add_layer(5, Some(Activation::ReLU));
    /// assert_eq!(network.layers.len(), 4);
    /// assert_eq!(network.layers[0].weights.shape(), (6, 4));
    /// assert_eq!(network.layers[3].weights.shape(), (3, 5));
    /// assert_eq!(network.layers[2].activation, Activation::Sigmoid);
    /// assert_eq!(network.layers[3].activation, Activation::ReLU);
    /// ```
    pub fn add_layer(&mut self, size: usize, activation: Option<Activation>) {
        let layer = Layer::new(
            self.layers.last().unwrap().weights.cols,
            size,
            &self.initializer,
            activation.unwrap_or(self.layers.last().unwrap().activation),
        );
        self.layers.push(layer);
    }

    /// Feeds the specified input through the network, returning the output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut network = Network::default(&[3, 4, 2, 2]);
    /// let inputs = tensor![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]];
    /// let output = network.feed_forward(&inputs);
    /// assert_eq!(output.shape(), (4, 2));
    /// ```
    pub fn feed_forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut output = inputs.clone();
        for layer in &mut self.layers {
            output = layer.feed_forward(&output);
        }
        output
    }

    /// Performs backpropagation on the network, using the specified outputs and targets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut network = Network::default(&[3, 4, 2, 3]);
    /// let inputs = tensor![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]];
    /// let targets = tensor![[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [1.1, 1.2, 1.3], [1.4, 1.5, 1.6]];
    /// assert_eq!(network.layers[0].inputs, None);
    /// assert_eq!(network.layers[0].output, None);
    /// network.feed_forward(&inputs);
    /// network.back_propagate(&targets);
    /// assert_ne!(network.layers[0].inputs, None);
    /// assert_ne!(network.layers[0].output, None);
    /// ```
    pub fn back_propagate(&mut self, targets: &Tensor) {
        let final_layer_output_shape = self.layers.last().unwrap().output.as_ref().unwrap().shape();
        if targets.shape() != final_layer_output_shape {
            panic!(
                "Target shape {:?} does not match the final layer's output shape {:?}",
                targets.shape(),
                final_layer_output_shape
            );
        }
        for layer in self.layers.iter_mut().rev() {
            layer.back_propagate(targets, &self.loss_function, &mut self.optimizer);
        }
    }

    /// Trains the network on the specified inputs and targets for the specified number of epochs.
    pub fn train(&mut self, inputs: &Tensor, targets: &Tensor, batch_size: usize, epochs: usize) {
        let num_batches = (inputs.rows as f64 / batch_size as f64).ceil() as usize;

        for epoch in 0..epochs {
            let mut error_sum = 0.0;

            for batch in 0..num_batches {
                let batch_start = batch * batch_size;
                let batch_end = (batch + 1) * batch_size;
                let inputs_batch = &inputs.slice(batch_start, batch_end);
                let targets_batch = &targets.slice(batch_start, batch_end);

                let outputs = self.feed_forward(&inputs_batch);
                let error = targets_batch.sub(&outputs);

                self.back_propagate(&targets_batch);

                error_sum += error.sum();
            }

            if epoch % 10 == 0 {
                let mse = error_sum / (inputs.rows as f64);
                println!("Epoch: {}, MSE: {}", epoch, mse);
            }
        }
    }
}
