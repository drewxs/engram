use crate::{
    activation::Activation, initializer::Initializer, layer::Layer, loss::mean_squared_error,
    optimizer::Optimizer, tensor::Tensor,
};

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    activation: Activation,
    #[allow(dead_code)]
    optimizer: Optimizer,
    learning_rate: f64,
}

impl Network {
    /// Creates a new `Network` with the specified layers, activation function, and learning rate.
    pub fn new(
        layer_sizes: Vec<usize>,
        initializer: Initializer,
        activation: Activation,
        optimizer: Optimizer,
        learning_rate: f64,
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
            learning_rate,
        }
    }

    /// Feeds the specified input through the network, returning the output.
    pub fn feed_forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut curr = inputs.clone();

        for layer in &mut self.layers {
            curr = layer.feed_forward(&curr, self.activation);
        }

        curr
    }

    /// Performs backpropagation on the network, using the specified outputs and targets.
    pub fn back_propagate(&mut self, outputs: &Tensor, targets: &Tensor) {
        let error = targets.sub(outputs);
        let mut delta = error.clone();

        for layer in self.layers.iter_mut().rev() {
            delta = layer.back_propagate(&mut delta, self.activation, self.learning_rate);
        }
    }

    /// Trains the network on the specified inputs and targets for the specified number of epochs.
    pub fn train(&mut self, inputs: &Tensor, targets: &Tensor, batch_size: usize, epochs: usize) {
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for batch_index in 0..inputs.shape().0 / batch_size {
                let start = batch_index * batch_size;
                let end = start + batch_size;
                let batch_inputs = inputs.slice(start, end);
                let batch_targets = targets.slice(start, end);

                let output = self.feed_forward(&batch_inputs);
                let loss = mean_squared_error(&output, &batch_targets);
                epoch_loss += loss;
            }

            let epoch_loss = epoch_loss / (inputs.shape().0 as f64);
            println!("Epoch {}: loss={}", epoch, epoch_loss);
        }
    }
}
