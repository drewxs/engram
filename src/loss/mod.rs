//! Loss functions.

mod bce;
mod mse;

pub use bce::*;
pub use mse::*;

use crate::Tensor;

/// Loss functions that can be used to train a neural network.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Loss {
    /// Binary cross-entropy.
    BCE,
    /// Mean squared error.
    MSE,
}

impl Loss {
    pub fn loss(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        match self {
            Loss::BCE => bce(predictions, targets),
            Loss::MSE => mse(predictions, targets),
        }
    }

    pub fn grad(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        match self {
            Loss::BCE => d_bce(predictions, targets),
            Loss::MSE => d_mse(predictions, targets),
        }
    }
}
