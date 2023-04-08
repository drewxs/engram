//! Implements different optimizers to be used with neural networks.
//!
//! Use to minimize the loss function during the training process of a neural network by
//! adjusting the weights of the network based on the gradients of the loss function with
//! respect to the weights.
//!
//! # Examples
//!
//! ```
//! use engram::{Tensor, optimizer::{Optimizer, SGD, Adagrad}};
//!
//! let optimizer = Optimizer::Adagrad(Adagrad::new(0.1, 0.01, Some(0.001), (4, 3));
//! optimizer.step(&mut layer.weights, &mut gradients);
//! ```

use crate::Tensor;

/// Stochastic Gradient Descent (SGD): A basic optimizer that updates the weights based on
/// the gradients of the loss function with respect to the weights multiplied by a learning rate.
#[derive(Debug)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn step(&self, weights: &mut Tensor, gradients: &Tensor) {
        weights.sub_assign(&gradients.mul_scalar(self.learning_rate));
    }
}

/// Adaptive Gradient (Adagrad): Adapts the learning rate based on the history of gradients.
/// Divides the learning rate by a running average of the magnitude of the gradients.
/// This allows the learning rate to decrease for parameters that have consistently large gradients
/// and increase for parameters that have consistently small gradients.
/// Includes an option to apply weight decay regularization to the gradients.
#[derive(Debug)]
pub struct Adagrad {
    learning_rate: f64,
    epsilon: f64,
    accumulators: Tensor,
    weight_decay: Option<f64>,
}

impl Adagrad {
    pub fn new(
        learning_rate: f64,
        epsilon: f64,
        weight_decay: Option<f64>,
        shape: (usize, usize),
    ) -> Self {
        Self {
            learning_rate,
            epsilon,
            accumulators: Tensor::zeros(shape.0, shape.1),
            weight_decay,
        }
    }

    pub fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor) {
        let sq_gradients = gradients.square();

        if let Some(wd) = self.weight_decay {
            gradients.sub_assign(&weights.mul_scalar(wd));
        }

        let mut new_accumulators = sq_gradients.add(&self.accumulators);
        self.accumulators.add_scalar_assign(self.epsilon);

        weights.sub_assign(
            &gradients
                .div(&(new_accumulators.mapv(&f64::sqrt)))
                .mul_scalar(self.learning_rate),
        );

        std::mem::swap(&mut self.accumulators, &mut new_accumulators);
    }
}

/// Optimizer enum that allows for different optimizers to be used with neural networks.
#[derive(Debug)]
pub enum Optimizer {
    SGD {
        learning_rate: f64,
    },
    Adagrad {
        learning_rate: f64,
        epsilon: f64,
        weight_decay: Option<f64>,
        shape: (usize, usize),
    },
}

impl Optimizer {
    pub fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor) {
        match self {
            Optimizer::SGD { learning_rate } => {
                let sgd = SGD::new(*learning_rate);
                sgd.step(weights, gradients);
            }
            Optimizer::Adagrad {
                learning_rate,
                epsilon,
                weight_decay,
                shape,
            } => {
                let mut adagrad = Adagrad::new(*learning_rate, *epsilon, *weight_decay, *shape);
                adagrad.step(weights, gradients);
            }
        }
    }
}
