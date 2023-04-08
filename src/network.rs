use crate::{Activation, Initializer, Layer, Optimizer, Tensor};

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    activation: Activation,
    optimizer: Optimizer,
}

impl Network {
    /// Creates a new `Network` with the specified layers, activation function, and learning rate.
    pub fn new(
        layer_sizes: Vec<usize>,
        initializer: Initializer,
        activation: Activation,
        optimizer: Optimizer,
    ) -> Network {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let layer = Layer::new(layer_sizes[i], layer_sizes[i + 1], &initializer);
            layers.push(layer);
        }

        Network {
            layers,
            activation,
            optimizer,
        }
    }

    /// Feeds the specified input through the network, returning the output.
    pub fn feed_forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut output = inputs.clone();
        for layer in &mut self.layers {
            output = layer.feed_forward(&output, self.activation);
        }
        output
    }

    /// Performs backpropagation on the network, using the specified outputs and targets.
    pub fn back_propagate(&mut self, outputs: &Tensor, targets: &Tensor) {
        let error = targets.sub(outputs);
        let mut delta = error.clone();

        for layer in self.layers.iter_mut().rev() {
            delta = layer.back_propagate(&mut delta, self.activation);
        }
    }

    /// Trains the network on the specified inputs and targets for the specified number of epochs.
    pub fn train(&mut self, inputs: &Tensor, targets: &Tensor, batch_size: usize, epochs: usize) {
        let num_batches = (inputs.shape().0 as f64 / batch_size as f64).ceil() as usize;

        for epoch in 0..epochs {
            let mut error_sum = 0.0;

            for batch in 0..num_batches {
                let batch_start = batch * batch_size;
                let batch_end = (batch + 1) * batch_size;
                let inputs_batch = inputs.slice(batch_start, batch_end);
                let targets_batch = targets.slice(batch_start, batch_end);

                let outputs = self.feed_forward(&inputs_batch);
                let mut error = targets_batch.sub(&outputs);

                self.back_propagate(&mut error, &targets_batch);

                for layer in &mut self.layers {
                    self.optimizer
                        .step(&mut layer.weights, &mut layer.weights_gradients);
                    self.optimizer
                        .step(&mut layer.biases, &mut layer.biases_gradients);
                }

                error_sum += error.sum();
            }

            if epoch % 10 == 0 {
                let mse = error_sum / (inputs.shape().0 as f64);
                println!("Epoch: {}, MSE: {}", epoch, mse);
            }
        }
    }
}
