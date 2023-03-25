use std::ops::{Add, Mul, Sub};

use crate::{activation::Activation, initializer::Initializer, matrix::Matrix};

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}

impl Network {
    pub fn new(
        layers: Vec<usize>,
        initializer: Initializer,
        activation: Activation,
        learning_rate: f64,
    ) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(initializer.initialize(layers[i], layers[i + 1]));
            biases.push(initializer.initialize(layers[i + 1], 1));
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

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        if inputs.shape()[1] != self.layers[0] {
            panic!("Invalid input size");
        }

        let mut current = inputs.reversed_axes();
        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .clone()
                .mul(&current)
                .add(&self.biases[i])
                .map(&|x: &f64| self.activation.apply(*x));
            self.data.push(current.clone());
        }

        current.to_owned()
    }

    pub fn back_propagate(&mut self, outputs: Matrix, targets: Matrix) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("Invalid target size");
        }

        let parsed = Matrix::from(outputs);
        let mut errors = Matrix::from(targets).sub(&parsed);
        let mut gradients = parsed.map(&|x: &f64| self.activation.gradient(*x));

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i]
                .clone()
                .add(&gradients.clone().mul(&self.data[i].clone().reversed_axes()));
            self.biases[i] = self.biases[i].clone().add(&gradients);
            errors = self.weights[i].clone().reversed_axes().mul(&errors);
            gradients = self.data[i].map(&|x: &f64| self.activation.gradient(*x));
        }
    }
}
