use crate::{
    activation::Activation,
    initializer::Initializer,
    tensor::{Tensor, Tensor1D, Tensor2D},
};

#[derive(Debug)]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    data: Vec<Tensor>,
    activation: Activation,
    learning_rate: f64,
}

impl Network {
    /// Creates a new `Network` with the specified number of layers, initializing technique, activation function, and learning rate.
    pub fn new(
        layers: Vec<usize>,
        initializer: Initializer,
        activation: Activation,
        learning_rate: f64,
    ) -> Network {
        let mut weights = Vec::with_capacity(layers.len() - 1);
        let mut biases = Vec::with_capacity(layers.len() - 1);

        for i in 0..layers.len() - 1 {
            let f_in = layers[i];
            let f_out = layers[i + 1];
            weights.push(Tensor::initialize(f_in, f_out, &initializer));
            biases.push(Tensor::initialize(f_out, 1, &initializer));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate,
        }
    }

    /// Feeds the specified input through the network, returning the output.
    pub fn feed_forward(&mut self, inputs: Tensor1D) -> Tensor1D {
        if inputs.len() != self.layers[0] {
            panic!("Invalid input size");
        }

        let mut curr = Tensor::from(vec![inputs]).transpose();
        self.data = vec![curr.clone()];

        for i in 0..self.layers.len() - 1 {
            curr = self.weights[i]
                .clone()
                .mul(&curr)
                .add(&self.biases[i])
                .map(&|x: f64| self.activation.apply(x));
            self.data.push(curr.clone());
        }

        curr.data[0].to_owned()
    }

    /// Performs backpropagation on the network, using the specified outputs and targets.
    pub fn back_propagate(&mut self, outputs: Tensor1D, targets: Tensor1D) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("Invalid target size");
        }

        let mut parsed = Tensor::from(vec![outputs]);
        let mut errors = Tensor::from(vec![targets]).sub(&parsed);
        let mut gradients = parsed.map(&|x: f64| self.activation.gradient(x));

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot(&errors).map(&|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.mul(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().mul(&errors);
            gradients = self.data[i].map(&|x: f64| self.activation.gradient(x));
        }
    }

    /// Trains the network on the specified inputs and targets for the specified number of epochs.
    pub fn train(&mut self, inputs: Tensor2D, targets: Tensor2D, epochs: usize) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }

            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone());
                self.back_propagate(outputs, targets[j].clone());
            }
        }
    }
}
