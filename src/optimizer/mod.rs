//! Optimization algorithms.
//!
//! Use to minimize the loss function during the training process of a neural network by
//! adjusting the weights of the network based on the gradients of the loss function with
//! respect to the weights.

mod adagrad;
mod sgd;

pub use adagrad::Adagrad;
pub use sgd::SGD;

use std::fmt::Debug;

use crate::Tensor;

pub trait Optimizer: Debug {
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor);
}
