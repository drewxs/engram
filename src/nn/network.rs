//! Neural Network module.
//!
//! This module provides the base building blocks for creating and training neural networks.

use crate::{Dataset, Loss, Optimizer, Regularization, Tensor};

use super::layer::Layer;

pub struct NeuralNetwork<L, O>
where
    L: Layer,
    O: Optimizer,
{
    pub layers: Vec<L>,
    pub optimizer: O,
    pub regularization: Option<Regularization>,
}

impl<L, O> NeuralNetwork<L, O>
where
    L: Layer,
    O: Optimizer + Default,
{
    pub fn new(optimizer: O, regularization: Option<Regularization>) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            optimizer,
            regularization,
        }
    }

    pub fn default() -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            optimizer: O::default(),
            regularization: None,
        }
    }

    pub fn add_layer(&mut self, layer: L) {
        self.layers.push(layer);
    }

    pub fn train(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.train();
        }
    }

    pub fn eval(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.eval();
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut layer_input = input.clone();
        for layer in self.layers.iter_mut() {
            let layer_output = layer.forward(&layer_input);
            layer_input = layer_output;
        }
        layer_input
    }

    pub fn backward(&mut self, target: &Tensor, loss_fn: &Loss) -> f64 {
        let mut loss = 0.0;
        for layer in self.layers.iter_mut().rev() {
            loss += layer.backward(&target, &loss_fn);
        }
        loss
    }

    pub fn fit(&mut self, dataset: &Dataset, loss_fn: &Loss, epochs: usize, batch_size: usize) {
        self.train();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let batches = dataset.batches(batch_size);

            for (input_batch, target_batch) in batches {
                self.forward(&input_batch);

                let loss = self.backward(&target_batch, loss_fn);
                total_loss += loss + self.regularize();

                self.step();
            }

            let avg_loss = total_loss / dataset.inputs.rows as f64;
            println!("Epoch {}: Average Loss = {}", epoch + 1, avg_loss);
        }
    }

    pub fn regularize(&mut self) -> f64 {
        if self.regularization.is_none() {
            return 0.0;
        }
        let reg = self.regularization.as_ref().unwrap();

        let loss = self
            .layers
            .iter()
            .fold(0.0, |acc, layer| acc + layer.regularization_loss(&reg));

        for layer in self.layers.iter_mut() {
            layer.apply_regularization(&reg);
        }

        loss
    }

    pub fn step(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.update_parameters(&mut self.optimizer);
        }
    }

    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        self.eval();
        self.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_xor() {
        let mut nn = NeuralNetwork::new(SGD { learning_rate: 0.1 }, Some(Regularization::L2(0.01)));
        nn.add_layer(DenseLayer::new(
            2,
            2,
            Initializer::Constant(0.0),
            Activation::Sigmoid,
        ));
        nn.add_layer(DenseLayer::new(
            2,
            1,
            Initializer::Constant(0.0),
            Activation::Sigmoid,
        ));

        let inputs = tensor![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = tensor![[0.], [1.], [1.], [0.]];

        let dataset = Dataset::new(inputs, targets);
        let loss_fn = Loss::MSE;
        let epochs = 1000;
        let batch_size = 1;

        nn.fit(&dataset, &loss_fn, epochs, batch_size);

        let batches = dataset.batches(batch_size);
        for (i, (input_batch, _)) in batches.iter().enumerate() {
            let predictions = nn.predict(&input_batch);
            let target = dataset.targets.data[i][0];
            println!("Predicted: {:?}, Target: {:.2}", predictions.data, target);
        }
        assert_eq!(0.0, 1.0);
    }

    // #[test]
    // fn test_1x1_constant_network() {
    //     // Test a simple network with a 1x1 layer and a 1x1 output.
    //     let mut nn = NeuralNetwork::default();
    //     nn.add_layer(NetworkLayer::FeedForward(Layer::default(1, 1)));
    //
    //     // The outputs are just 1 times the input plus 1, so the goal is for the
    //     // network to learn the weights [[1.0]] and bias [1.0].
    //     let inputs = tensor![[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]];
    //     let targets = tensor![[2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.], [11.]];
    //
    //     let loss_fn = Loss::MSE;
    //     let epochs = 1000;
    //
    //     nn.train(&inputs, &targets, &loss_fn, epochs);
    //
    //     let x = tensor![5.];
    //     let y_true = 6.;
    //     let y_pred = nn.predict(&x);
    //
    //     dbg!(&nn.layers);
    //     println!("Predicted: {:.2}, Expected: {:.2}", y_pred, y_true);
    //
    //     assert!((y_true - y_pred).abs() < 0.5);
    // }
}
