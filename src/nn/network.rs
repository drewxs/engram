//! Neural Network module.
//!
//! This module provides the base building blocks for creating and training neural networks.

use crate::{Dataset, Layer, Loss, Optimizer, Regularization, Tensor};

#[derive(Debug)]
pub struct Network<L, O, R>
where
    L: Layer,
    O: Optimizer,
    R: Regularization,
{
    pub layers: Vec<L>,
    pub optimizer: O,
    pub regularization: R,
}

impl<L, O, R> Network<L, O, R>
where
    L: Layer,
    O: Optimizer + Default,
    R: Regularization,
{
    pub fn new(optimizer: O, regularization: R) -> Self {
        Network {
            layers: Vec::new(),
            optimizer,
            regularization,
        }
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
        let loss = self.layers.iter().fold(0.0, |acc, layer| {
            acc + layer.regularization_loss(&self.regularization)
        });

        for layer in self.layers.iter_mut() {
            layer.apply_regularization(&self.regularization);
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
    fn test_1x1_constant_network() {
        // Test a simple network with a 1x1 layer and a 1x1 output.
        let mut model = Network::new(SGD::new(0.0), L1(0.0));
        model.layers.push(DenseLayer::new(
            1,
            1,
            Initializer::Constant(1.0),
            Activation::ReLU,
        ));

        dbg!(&model.optimizer);

        // The outputs are just 1 times the input plus 1, so the goal is for the
        // network to learn the weights [[1.0]] and bias [1.0].
        let inputs = tensor![[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]];
        let targets = tensor![[2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.], [11.]];

        let dataset = Dataset::new(inputs, targets);
        let loss_fn = Loss::MSE;
        let epochs = 1;
        let batch_size = 1;

        model.fit(&dataset, &loss_fn, epochs, batch_size);

        let weights = model.layers[0].weights().clone().data;
        let biases = model.layers[0].biases().clone().data;
        dbg!(weights, biases);

        let batches = dataset.batches(batch_size);
        for (i, (input_batch, _)) in batches.iter().enumerate() {
            let predictions = model.predict(&input_batch);
            let y_pred = predictions.data[0][0];
            let y_true = dataset.targets.data[i][0];

            println!("Predicted: {:.2?}, Target: {:.2}", y_pred, y_true);
            assert_eq!(y_pred, y_true);
        }
    }

    #[test]
    fn test_xor() {
        let mut model = Network::new(SGD::new(0.01), L1(0.01));
        model.layers = vec![
            DenseLayer::new(2, 2, Initializer::Xavier, Activation::Sigmoid),
            DenseLayer::new(2, 1, Initializer::Xavier, Activation::Sigmoid),
        ];

        let inputs = tensor![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = tensor![[0.], [1.], [1.], [0.]];

        let dataset = Dataset::new(inputs, targets);
        let loss_fn = Loss::MSE;
        let epochs = 30;
        let batch_size = 1;

        model.fit(&dataset, &loss_fn, epochs, batch_size);

        let batches = dataset.batches(batch_size);
        for (i, (input_batch, _)) in batches.iter().enumerate() {
            let predictions = model.predict(&input_batch);
            let y_pred = (predictions.data[0][0] + 0.5).floor();
            let y_true = dataset.targets.data[i][0];

            println!(
                "Predicted: {:.2?}, Target: {:.2}",
                predictions.data[0][0], y_true
            );
            assert_eq!(y_pred, y_true, "Failed at input {:?}", input_batch.data);
        }

        // assert_eq!(true, false);
    }
}
