//! Create and manipulate neural network layers.
//!
//! This module provides a Layer struct for representing a single layer in a neural network,
//! along with methods for feeding inputs through the layer and performing backpropagation.

mod dense;

pub use dense::DenseLayer;

use std::fmt::Debug;

use crate::{nn::Regularization, Loss, Optimizer, Tensor};

pub trait Layer: Debug {
    fn weights(&self) -> &Tensor;
    fn biases(&self) -> &Tensor;
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, target: &Tensor, loss_fn: &Loss) -> f64;
    fn update_parameters(&mut self, optimizer: &mut dyn Optimizer);
    fn regularization_loss(&self, reg: &dyn Regularization) -> f64;
    fn apply_regularization(&mut self, reg: &dyn Regularization);
    fn eval(&mut self);
    fn train(&mut self);
}
