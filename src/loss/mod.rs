//! Loss functions.

mod bce;
mod mse;

pub use bce::*;
pub use mse::*;

use crate::Tensor;

/// Loss functions that can be used to train a neural network.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Loss {
    BinaryCrossEntropy,
    MeanSquaredError,
}

impl Loss {
    pub fn loss(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        match self {
            Loss::BinaryCrossEntropy => bce(predictions, targets),
            Loss::MeanSquaredError => mse(predictions, targets),
        }
    }

    pub fn gradient(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        match self {
            Loss::BinaryCrossEntropy => d_bce(predictions, targets),
            Loss::MeanSquaredError => d_mse(predictions, targets),
        }
    }
}
