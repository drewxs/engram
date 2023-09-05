//! Loss functions.

pub mod bce;
pub mod mse;

pub use bce::*;
pub use mse::*;

use crate::Tensor;

pub enum LossFunction {
    BinaryCrossEntropy,
    MeanSquaredError,
}

pub trait Loss {
    fn loss(&self, predictions: &Tensor, targets: &Tensor) -> f64;
}

impl Loss for LossFunction {
    fn loss(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        match self {
            LossFunction::BinaryCrossEntropy => binary_cross_entropy(predictions, targets),
            LossFunction::MeanSquaredError => mean_squared_error(predictions, targets),
        }
    }
}
