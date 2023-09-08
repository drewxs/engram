//! Loss functions.

mod bce;
mod mse;

pub use bce::*;
pub use mse::*;

use crate::Tensor;

/// Loss functions that can be used to train a neural network.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum LossFunction {
    BinaryCrossEntropy,
    MeanSquaredError,
}

impl LossFunction {
    pub fn loss(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        match self {
            LossFunction::BinaryCrossEntropy => binary_cross_entropy(predictions, targets),
            LossFunction::MeanSquaredError => mean_squared_error(predictions, targets),
        }
    }
}
