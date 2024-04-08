use crate::{Dataset, Loss, Optimizer, Regularization, Tensor, SGD};

use super::layer::Layer;

#[derive(Debug, Clone)]
pub enum NetworkLayer {
    FeedForward(Layer),
}

pub enum Mode {
    Train,
    Eval,
}

pub struct NeuralNetwork {
    pub layers: Vec<NetworkLayer>,
    pub optimizer: Optimizer,
    pub regularization: Option<Regularization>,
    mode: Mode,
}

impl NeuralNetwork {
    pub fn new(optimizer: Optimizer, regularization: Option<Regularization>) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            optimizer,
            regularization,
            mode: Mode::Train,
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

    pub fn train(&mut self) {
        self.mode = Mode::Train;
    }

    pub fn eval(&mut self) {
        self.mode = Mode::Eval;
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut layer_input = input.clone();
        for layer in self.layers.iter_mut() {
            match layer {
                NetworkLayer::FeedForward(l) => {
                    let layer_output = l.forward(&layer_input);
                    if let Mode::Train = self.mode {
                        layer_input = layer_output;
                    }
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
        let mut total_loss = 0.0;
        if let Some(reg) = &self.regularization {
            for layer in self.layers.iter_mut() {
                match layer {
                    NetworkLayer::FeedForward(l) => {
                        let reg_loss = reg.loss(&l.weights);
                        total_loss += reg_loss;

                        let reg_grad = reg.grad(&l.weights);
                        if let Some(dw) = &l.d_weights {
                            l.d_weights = Some(dw + &reg_grad);
                        }
                    }
                }
            }
        }
        total_loss
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
        let mut nn = NeuralNetwork::new(
            Optimizer::SGD(SGD { learning_rate: 0.1 }),
            Some(Regularization::L2(0.01)),
        );
        nn.add_layer(NetworkLayer::FeedForward(Layer::new(
            2,
            2,
            Initializer::Xavier,
            Activation::Sigmoid,
        )));
        nn.add_layer(NetworkLayer::FeedForward(Layer::new(
            2,
            1,
            Initializer::Xavier,
            Activation::Sigmoid,
        )));

        let inputs = tensor![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = tensor![[0.], [1.], [1.], [0.]];

        let dataset = Dataset::new(inputs, targets);
        let loss_fn = Loss::MSE;
        let epochs = 100;
        let batch_size = 1;

        nn.fit(&dataset, &loss_fn, epochs, batch_size);

        let predictions = nn.predict(&dataset.inputs);
        for (i, prediction) in predictions.iter_rows().enumerate() {
            let target = dataset.targets.data[i][0];
            let predicted = if prediction[0] >= 0.5 { 1.0 } else { 0.0 };

            assert_eq!(predicted, target);
        }
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
