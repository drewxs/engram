use crate::{Loss, Optimizer, Regularization, Tensor, SGD};

use super::layer::Layer;

#[derive(Debug)]
pub enum NetworkLayer {
    FeedForward(Layer),
}

pub struct NeuralNetwork {
    layers: Vec<NetworkLayer>,
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
            Optimizer::SGD(SGD { learning_rate: 0.1 }),
            Some(Regularization::L2(0.01)),
        )
    }

    pub fn add_layer(&mut self, layer: NetworkLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.layers
            .iter()
            .fold(input.clone(), |acc, layer| match layer {
                NetworkLayer::FeedForward(l) => l.forward(&acc),
            })
    }

    pub fn backward(
        &mut self,
        input: &Tensor,
        target: &Tensor,
        loss: &Loss,
    ) -> Vec<(Tensor, Tensor)> {
        let output = self.forward(input);
        let mut gradient = loss.gradient(&output, target);
        let mut gradients = Vec::new();

        for i in (0..self.layers.len()).rev() {
            let (d_input, mut d_weight, d_bias) = match &mut self.layers[i] {
                NetworkLayer::FeedForward(l) => {
                    let (d_input, mut d_weight, mut d_bias) = l.backward(input, &gradient);
                    if let Some(reg) = &self.regularization {
                        d_weight += reg.grad(&l.weights);
                        d_bias += reg.grad(&l.biases);
                    }
                    (d_input, d_weight, d_bias)
                }
            };

            if let Some(reg) = &self.regularization {
                d_weight += reg.grad(&d_weight);
            }

            gradients.push((d_weight, d_bias));

            if i > 0 {
                let next_layer = &self.layers[i - 1];
                match next_layer {
                    NetworkLayer::FeedForward(next_l) => {
                        if d_input.is_matmul_compatible(&next_l.weights) {
                            gradient = d_input.matmul(&next_l.weights);
                        } else {
                            gradient = d_input.matmul(&next_l.weights.transpose());
                        }
                    }
                }
            }
        }

        gradients
    }

    pub fn step(&mut self, gradients: &[(Tensor, Tensor)]) {
        for (layer, (d_weight, d_bias)) in self.layers.iter_mut().zip(gradients.iter()) {
            match layer {
                NetworkLayer::FeedForward(l) => {
                    self.optimizer.step(&mut l.weights, d_weight);
                    self.optimizer.step(&mut l.biases, d_bias);
                }
            }
        }
    }

    pub fn train(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        loss: &Loss,
        epochs: usize,
        batch_size: usize,
    ) -> (f64, f64, f64) {
        let mut total_loss = 0.0;
        let mut avg_loss = 0.0;
        let mut curr_loss = 0.0;

        let num_samples = inputs.rows;
        let num_batches = (num_samples + batch_size - 1) / batch_size;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for batch in 0..num_batches {
                let batch_start = batch * batch_size;
                let batch_end = usize::min(batch_start + batch_size, num_samples);

                let batch_inputs = inputs.slice(batch_start, batch_end);
                let batch_targets = targets.slice(batch_start, batch_end);

                let output = self.forward(&batch_inputs);
                let gradients = self.backward(&batch_inputs, &batch_targets, loss);
                let mut batch_loss = loss.loss(&output, &batch_targets).mean();

                if let Some(reg) = &self.regularization {
                    for layer in &self.layers {
                        match layer {
                            NetworkLayer::FeedForward(l) => batch_loss += reg.loss(&l.weights),
                        }
                    }
                }

                self.step(&gradients);

                epoch_loss += batch_loss;
            }

            curr_loss = epoch_loss / num_batches as f64;
            total_loss += curr_loss;

            if epoch != 0 && epoch % 10 == 0 {
                avg_loss = total_loss / epoch as f64;
                println!(
                    "Epoch {}: Total Loss: {:.4}, Avg Loss: {:.4}, Loss: {:.4}",
                    epoch, total_loss, avg_loss, curr_loss
                );
            }
        }

        (total_loss, avg_loss, curr_loss)
    }

    pub fn train_no_batch(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        loss: &Loss,
        epochs: usize,
    ) -> (f64, f64, f64) {
        let mut total_loss = 0.0;
        let mut avg_loss = 0.0;
        let mut epoch_loss = 0.0;

        for epoch in 0..epochs {
            let output = self.forward(&inputs);
            let gradients = self.backward(&inputs, &targets, loss);
            epoch_loss = loss.loss(&output, &targets).mean();

            if let Some(reg) = &self.regularization {
                for layer in &self.layers {
                    match layer {
                        NetworkLayer::FeedForward(l) => epoch_loss += reg.loss(&l.weights),
                    }
                }
            }

            total_loss += epoch_loss;

            self.step(&gradients);

            if epoch != 0 && epoch % 10 == 0 {
                avg_loss = total_loss / epoch as f64;
                println!(
                    "Epoch {}: Total Loss: {:.4}, Avg Loss: {:.4}, Loss: {:.4}",
                    epoch, total_loss, avg_loss, epoch_loss
                );
            }
        }

        (total_loss, avg_loss, epoch_loss)
    }

    pub fn predict(&mut self, input: &[f64]) -> f64 {
        let inputs = Tensor::from(vec![input.to_vec()]);
        let output = self.forward(&inputs);

        output.data[0][0]
    }

    pub fn predict_avg(&mut self, input: &[f64]) -> f64 {
        let inputs = Tensor::from(vec![input.to_vec()]);
        let output = self.forward(&inputs);

        output.mean()
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
            Activation::Sigmoid,
        )));
        nn.add_layer(NetworkLayer::FeedForward(Layer::new(
            2,
            1,
            Initializer::Kaiming,
            Activation::Sigmoid,
        )));

        let inputs = tensor![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = tensor![[0.], [1.], [1.], [0.]];
        let (_, _, loss) = nn.train(&inputs, &targets, &Loss::MeanSquaredError, 200, 4);
        println!("Loss: {}", loss);
        assert!(loss <= 0.5);

        let prediction = nn.predict_avg(&[1., 0.]);
        println!("Prediction: {}", prediction);
        assert!(prediction > 0.5);
    }

    #[test]
    fn test_1x1_constant_network() {
        // Test a simple network with a 1x1 layer and a 1x1 output.
        let mut nn = NeuralNetwork::default();
        nn.add_layer(NetworkLayer::FeedForward(Layer::new(
            1,
            1,
            Initializer::Constant(1.),
            Activation::ReLU,
        )));
        // The outputs are just 1 times the input plus 1, so the goal is for the
        // network to learn the weights [[1.0]] and bias [1.0].
        let inputs = tensor![[1.], [2.], [3.], [4.]];
        let targets = tensor![[2.], [3.], [4.], [5.]];

        // nn.train(&inputs, &targets, &Loss::MeanSquaredError, 100, 4);
        nn.train_no_batch(&inputs, &targets, &Loss::MeanSquaredError, 100);
        let prediction = nn.predict(&[5.]);
        let expected = 6.;

        println!("Predicted: {:.2}, Expected: {:.2}", prediction, expected);
        assert!((expected - prediction).abs() < 0.1);
    }
}
