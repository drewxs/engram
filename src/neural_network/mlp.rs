//! Multi-layer perceptron (MLP) neural network.
//!
//! A generic feed forward neural network (FNN), also known as a multi-layer perceptron (MLP).
//! Typically used for classification and regression tasks.

use crate::{Activation, Initializer, Layer, Loss, Optimizer, Tensor};

#[derive(Debug)]
pub struct Network {
    /// The layers in the network.
    pub layers: Vec<Layer>,
    /// The loss function used to train the network.
    pub loss_function: Loss,
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
    ///
    /// let network = Network::new(
    ///     &[6, 4, 1],
    ///     Initializer::Xavier,
    ///     Activation::ReLU,
    ///     Loss::MeanSquaredError,
    ///     Optimizer::Adagrad {
    ///         learning_rate: 0.1,
    ///         shape: (3, 1),
    ///         weight_decay: Some(0.01),
    ///         epsilon: Some(1e-8),
    ///     },
    /// );
    ///
    /// assert_eq!(network.layers.len(), 2);
    /// assert_eq!(network.layers[0].weights.shape(), (6, 4));
    /// assert_eq!(network.layers[1].weights.shape(), (4, 1));
    /// assert_eq!(network.layers[0].activation, Activation::ReLU);
    /// assert_eq!(network.loss_function, Loss::MeanSquaredError);
    /// assert_eq!(network.optimizer, Optimizer::Adagrad {
    ///    learning_rate: 0.1,
    ///    shape: (3, 1),
    ///    weight_decay: Some(0.01),
    ///    epsilon: Some(1e-8),
    /// })
    /// ```
    pub fn new(
        layer_sizes: &[usize],
        initializer: Initializer,
        activation: Activation,
        loss_function: Loss,
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
    ///
    /// let network = Network::default(&[6, 4, 2, 3]);
    ///
    /// assert_eq!(network.layers.len(), 3);
    /// assert_eq!(network.layers[0].weights.shape(), (6, 4));
    /// assert_eq!(network.layers[1].weights.shape(), (4, 2));
    /// assert_eq!(network.layers[2].weights.shape(), (2, 3));
    /// assert_eq!(network.layers[0].activation, Activation::Sigmoid);
    /// assert_eq!(network.loss_function, Loss::MeanSquaredError);
    /// assert_eq!(network.optimizer, Optimizer::SGD { learning_rate: 0.1 })
    /// ```
    pub fn default(layer_sizes: &[usize]) -> Network {
        Network::new(
            layer_sizes,
            Initializer::Xavier,
            Activation::Sigmoid,
            Loss::MeanSquaredError,
            Optimizer::SGD { learning_rate: 0.1 },
        )
    }

    /// Adds a layer to the network.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut network = Network::default(&[6, 4, 2, 3]);
    ///
    /// assert_eq!(network.layers.len(), 3);
    ///
    /// network.add_layer(5, Some(Activation::ReLU));
    ///
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
    ///
    /// let mut network = Network::default(&[3, 4, 2, 2]);
    /// let inputs = tensor![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]];
    /// let output = network.feed_forward(&inputs);
    ///
    /// assert_eq!(output.shape(), (4, 2));
    /// ```
    pub fn feed_forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut output = inputs.clone();
        for layer in &mut self.layers {
            output = layer.feed_forward(&output);
            println!("Output: {:?}", output.data);
        }
        output
    }

    /// Feeds the specified input through the network, updating the layer's inputs and output.
    /// This method is used when training the network.
    /// If you are only evaluating the network, use `feed_forward` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut network = Network::default(&[4, 3]);
    /// let inputs = tensor![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]];
    /// network.feed_forward_mut(&inputs);
    ///
    /// assert_ne!(network.layers.first().unwrap().inputs, None);
    /// ```
    pub fn feed_forward_mut(&mut self, inputs: &Tensor) {
        for layer in &mut self.layers {
            layer.feed_forward_mut(&inputs);
            println!("Output: {:?}", layer.output.as_ref().unwrap().data);
        }
    }

    /// Performs backpropagation on the network, using the specified outputs and targets.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut network = Network::default(&[3, 4, 2, 3]);
    ///
    /// let inputs = tensor![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]];
    /// let targets = tensor![[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [1.1, 1.2, 1.3], [1.4, 1.5, 1.6]];
    ///
    /// assert_eq!(network.layers[0].inputs, None);
    /// assert_eq!(network.layers[0].output, None);
    ///
    /// network.feed_forward(&inputs);
    /// network.back_propagate(&targets);
    ///
    /// assert_ne!(network.layers[0].inputs, None);
    /// assert_ne!(network.layers[0].output, None);
    /// ```
    pub fn back_propagate(&mut self, targets: &Tensor) -> f64 {
        let final_layer_output_shape = self.layers.last().unwrap().output.as_ref().unwrap().shape();
        if targets.shape() != final_layer_output_shape {
            panic!(
                "Target shape {:?} does not match the final layer's output shape {:?}",
                targets.shape(),
                final_layer_output_shape
            );
        }
        let mut loss = 0.0;
        for layer in self.layers.iter_mut().rev() {
            loss += layer.back_propagate(targets, &self.loss_function, &mut self.optimizer);
        }
        loss
    }

    /// Trains the network on the specified inputs and targets for the specified number of epochs.
    ///
    /// # Examples
    ///
    /// Training XOR:
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut network = Network::new(
    ///     &[2, 3, 1],
    ///     Initializer::Xavier,
    ///     Activation::ReLU,
    ///     Loss::MeanSquaredError,
    ///     Optimizer::SGD { learning_rate: 0.1 },
    /// );
    ///
    /// let inputs = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    /// let targets = tensor![[0.0], [1.0], [1.0], [0.0]];
    /// network.train(&inputs, &targets, 4, 100);
    ///
    /// for layer in network.layers {
    ///     assert_ne!(layer.inputs, None);
    ///     assert_ne!(layer.output, None);
    /// }
    /// ```
    pub fn train(&mut self, inputs: &Tensor, targets: &Tensor, batch_size: usize, epochs: usize) {
        if targets.cols != self.layers.last().unwrap().weights.cols {
            panic!(
                "Target cols {:?} does not match the final layer's output cols {:?}",
                targets.cols,
                self.layers.last().unwrap().weights.cols
            );
        }

        let num_batches = (inputs.rows as f64 / batch_size as f64).ceil() as usize;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for batch in 0..num_batches - 1 {
                let batch_start = batch * batch_size;
                let batch_end = (batch + 1) * batch_size;
                let inputs_batch = &inputs.slice(batch_start, batch_end);
                let targets_batch = &targets.slice(batch_start, batch_end);

                self.feed_forward_mut(&inputs_batch);
                let loss = self.back_propagate(&targets_batch);

                total_loss += loss;
            }

            if epoch % 10 == 0 {
                let avg_loss = total_loss / (inputs.rows as f64);
                println!("Epoch: {}, Avg loss: {}", epoch, avg_loss);
            }
        }
    }

    /// Predicts the output for the specified inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut network = Network::new(
    ///     &[2, 3, 3, 1],
    ///     Initializer::Xavier,
    ///     Activation::ReLU,
    ///     LossFunction::MeanSquaredError,
    ///     Optimizer::SGD { learning_rate: 0.1 },
    /// );
    ///
    /// let inputs = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    /// let targets = tensor![[0.0], [1.0], [1.0], [0.0]];
    ///
    /// network.train(&inputs, &targets, 4, 100);
    ///
    /// let output = network.predict(&[1.0, 0.0]);
    /// let expected = 1.0;
    /// let prediction = output.data[0][0];
    ///
    /// println!("Predicted: {:.2}, Expected: {:.2}", prediction, expected);
    /// // TODO: This is not working, the prediction is always 0.0 or close to it.
    /// //       Not sure if this is a calculation error with the optimizer or loss function,
    /// //       or just a hyperparameter tuning problem
    /// // assert!((expected - prediction).abs() < 0.1);
    /// ```
    pub fn predict(&mut self, inputs: &[f64]) -> Tensor {
        let inputs = Tensor::from(vec![inputs.to_vec()]);
        let output = self.feed_forward(&inputs);

        output
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_1x1_constant_network() {
        // Test a simple network with a 1x1 layer and a 1x1 output.
        let mut network = Network::new(
            &[1, 1],
            Initializer::Constant(1.),
            Activation::ReLU,
            Loss::MeanSquaredError,
            Optimizer::SGD { learning_rate: 0.1 },
        );
        // The outputs are just 1 times the input plus 1, so the goal is for the
        // network to learn the weights [[1.0]] and bias [1.0].
        let inputs = tensor![[0.], [1.], [2.], [3.]];
        let targets = tensor![[1.], [2.], [3.], [4.]];

        network.train(&inputs, &targets, 4, 10);
        let output = network.predict(&[4.]);
        let expected = 5.;
        let prediction = output.data[0][0];

        println!("Predicted: {:.2}, Expected: {:.2}", prediction, expected);
        assert!((expected - prediction).abs() < 0.1);
    }

    #[test]
    fn test_sum_network() {
        // Test a simple network with a 4x4 layer and a 4x4 output.
        let mut network = Network::new(
            &[4, 1],
            Initializer::Kaiming,
            Activation::Sigmoid,
            Loss::MeanSquaredError,
            Optimizer::SGD {
                learning_rate: 0.01,
            },
        );

        let inputs = tensor![
            [0., 1., 2., 3.],
            [1., 2., 3., 4.],
            [2., 3., 4., 5.],
            [3., 4., 5., 6.],
            [8., 1., 6., 3.],
            [1., 2., 1., 1.],
            [9., 8., 8., 0.],
            [1., 2., 3., 4.]
        ];
        let targets = tensor![[6.], [10.], [14.], [18.], [18.], [5.], [25.], [10.]];

        network.train(&inputs, &targets, 4, 10);
        let prediction = network.predict(&[6., 7., 8., 9.]).data[0][0];
        let expected = 22.;

        println!("Predicted: {:.2}, Expected: {:.2}", prediction, expected);
        assert!((expected - prediction).abs() < 0.1);
    }

    // #[test]
    // fn test_xavier_sigmoid_network() {
    //     // Again test a network with a 1x1 layer and a 1x1 output, but this time
    //     // we want the weight to stay at 0 and the bias to increase to 1.
    //     let mut network = Network::new(
    //         &[1, 1],
    //         Initializer::Xavier,
    //         Activation::Sigmoid,
    //         Loss::MeanSquaredError,
    //         Optimizer::SGD { learning_rate: 0.1 },
    //     );
    //     // The outputs are just 1 times the input plus 0.
    //     let inputs = tensor![[0.], [1.], [2.], [3.]];
    //     let targets = tensor![[1.], [1.], [1.], [1.]];
    //
    //     network.train(&inputs, &targets, 4, 10);
    //     let output = network.predict(&[4.]);
    //     let expected = 1.;
    //     let prediction = output.data[0][0];
    //
    //     println!("Predicted: {:.2}, Expected: {:.2}", prediction, expected);
    //     assert!((expected - prediction).abs() < 0.1);
    // }
}
