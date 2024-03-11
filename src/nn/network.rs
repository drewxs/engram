use crate::{Loss, Optimizer, Regularization, Tensor, SGD};

use super::layer::Layer;

#[derive(Debug, Clone)]
pub enum NetworkLayer {
    FeedForward(Layer),
}

pub struct NeuralNetwork {
    pub layers: Vec<NetworkLayer>,
    pub optimizer: Optimizer,
    pub regularization: Option<Regularization>,
}

impl NeuralNetwork {
    pub fn new(optimizer: Optimizer, regularization: Option<Regularization>) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            optimizer,
            regularization,
        }
    }

    pub fn default() -> Self {
        NeuralNetwork::new(
            Optimizer::SGD(SGD {
                learning_rate: 0.01,
            }),
            Some(Regularization::L2(0.01)),
        )
    }

    pub fn add_layer(&mut self, layer: NetworkLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut layer_input = input.clone();

        for layer in self.layers.iter_mut() {
            match layer {
                NetworkLayer::FeedForward(l) => {
                    let layer_output = l.forward(&layer_input);
                    layer_input = layer_output;
                }
            }
        }

        layer_input
    }

    pub fn backward(&mut self, target: &Tensor, loss_fn: &Loss) -> f64 {
        let mut loss = 0.0;
        for layer in self.layers.iter_mut().rev() {
            match layer {
                NetworkLayer::FeedForward(l) => {
                    loss += l.backward(&target, &loss_fn);
                }
            };
        }
        loss
    }

    pub fn step(&mut self) {
        for layer in self.layers.iter_mut() {
            match layer {
                NetworkLayer::FeedForward(l) => {
                    if let Some(d_weights) = &l.d_weights {
                        self.optimizer.step(&mut l.weights, d_weights);
                    }
                    if let Some(d_biases) = &l.d_biases {
                        self.optimizer.step(&mut l.biases, d_biases);
                    }
                }
            }
        }
    }

    pub fn train(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        loss_fn: &Loss,
        epochs: usize,
    ) -> f64 {
        let mut total_loss = 0.0;

        for epoch in 1..=epochs {
            self.forward(&inputs);
            let mut loss = self.backward(&targets, &loss_fn);
            total_loss += loss;

            if let Some(reg) = &self.regularization {
                for layer in &self.layers {
                    match layer {
                        NetworkLayer::FeedForward(l) => loss += reg.loss(&l.weights),
                    }
                }
            }

            self.step();

            if epoch % (epochs / 10) == 0 {
                println!("Epoch {}, Loss: {}", epoch, loss);
            }
        }

        total_loss
    }

    pub fn predict(&mut self, input: &Tensor) -> f64 {
        let output = self.forward(&input);
        output.data[0][0]
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor, Activation, Initializer};

    use super::*;

    #[test]
    fn test_xor() {
        let mut nn = NeuralNetwork::default();
        nn.add_layer(NetworkLayer::FeedForward(Layer::new(
            2,
            2,
            Initializer::Kaiming,
            Activation::ReLU,
        )));
        nn.add_layer(NetworkLayer::FeedForward(Layer::new(
            2,
            1,
            Initializer::Kaiming,
            Activation::ReLU,
        )));
        // nn.add_layer(NetworkLayer::FeedForward(Layer::default(2, 2)));
        // nn.add_layer(NetworkLayer::FeedForward(Layer::default(2, 1)));

        let inputs = tensor![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = tensor![[0.], [1.], [1.], [0.]];

        let loss_fn = Loss::MSE;
        let epochs = 1000;

        let total_loss = nn.train(&inputs, &targets, &loss_fn, epochs);
        let avg_loss = total_loss / epochs as f64;

        println!("Avg loss: {}", avg_loss);

        let x = tensor![[1., 0.]];
        let y_true = 1.;
        let y_pred = nn.predict(&x);
        println!("Predicted: {:.4}, Expected: {:.4}", y_pred, y_true);
        assert!((y_pred - y_true).abs() < 0.1);

        let x = tensor![[1., 1.]];
        let y_true = 0.;
        let y_pred = nn.predict(&x);
        println!("Predicted: {:.4}, Expected: {:.4}", y_pred, y_true);
        assert!((y_pred - y_true).abs() < 0.1);
    }

    #[test]
    fn test_1x1_constant_network() {
        // Test a simple network with a 1x1 layer and a 1x1 output.
        let mut nn = NeuralNetwork::default();
        nn.add_layer(NetworkLayer::FeedForward(Layer::default(1, 1)));

        // The outputs are just 1 times the input plus 1, so the goal is for the
        // network to learn the weights [[1.0]] and bias [1.0].
        let inputs = tensor![[1.], [2.], [3.], [4.]];
        let targets = tensor![[2.], [3.], [4.], [5.]];

        let loss_fn = Loss::MSE;
        let epochs = 1000;

        nn.train(&inputs, &targets, &loss_fn, epochs);

        let x = tensor![[5.]];
        let y_true = 6.;
        let y_pred = nn.predict(&x);

        dbg!(&nn.layers);
        println!("Predicted: {:.2}, Expected: {:.2}", y_pred, y_true);

        // assert!((y_true - y_pred).abs() < 1.0);
    }
}
