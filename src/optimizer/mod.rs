//! Optimization algorithms.
//!
//! Use to minimize the loss function during the training process of a neural network by
//! adjusting the weights of the network based on the gradients of the loss function with
//! respect to the weights.

mod adagrad;
mod sgd;

pub use adagrad::Adagrad;
pub use sgd::SGD;

use crate::Tensor;

/// Optimizer enum that allows for different optimizers to be used with neural networks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    /// Stochastic Gradient Descent (SGD) optimizer.
    SGD { learning_rate: f64 },
    /// An adaptive gradient descent optimizer.
    Adagrad {
        learning_rate: f64,
        shape: (usize, usize),
        weight_decay: Option<f64>,
        epsilon: Option<f64>,
    },
}

pub trait Optimize {
    fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor);
}

impl Optimize for Optimizer {
    fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor) {
        match self {
            Optimizer::SGD { learning_rate } => {
                let mut optimizer = SGD::new(*learning_rate);
                optimizer.step(weights, gradients);
            }
            Optimizer::Adagrad {
                learning_rate,
                weight_decay,
                epsilon,
                ..
            } => {
                let mut optimizer =
                    Adagrad::new(*learning_rate, weights.shape(), *weight_decay, *epsilon);
                optimizer.step(weights, gradients);
            }
        }
    }
}
