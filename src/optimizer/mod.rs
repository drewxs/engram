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
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Stochastic Gradient Descent (SGD) optimizer.
    SGD(SGD),
    /// An adaptive gradient descent optimizer.
    Adagrad(Adagrad),
}

pub trait Optimize {
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor);
}

impl Optimizer {
    /// Updates the weights based on the gradients of the loss function
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    ///
    /// let mut optimizer = Optimizer::SGD(SGD { learning_rate: 0.1 });
    /// let mut weights = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let mut gradients = weights.grad(&Activation::ReLU);
    ///
    /// optimizer.step(&mut weights, &mut gradients);
    ///
    /// # assert_eq!(weights, tensor![[0.9, 1.9, 2.9], [3.9, 4.9, 5.9]]);
    /// ```
    pub fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) {
        match self {
            Optimizer::SGD(optimizer) => optimizer.step(weights, gradients),
            Optimizer::Adagrad(optimizer) => optimizer.step(weights, gradients),
        }
    }
}
