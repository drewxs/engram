//! Mean squared error loss.

use crate::Tensor;

/// Returns mean squared error between the predictions and the targets.
///
/// # Examples
///
/// ```
/// # use engram::{tensor, loss::mse};
/// let predictions = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let targets = tensor![[1.2, 2.4, 3.1], [4.4, 4.7, 5.9]];
/// let loss = mse(&predictions, &targets);
/// # assert_eq!(loss, tensor![[0.03999999999999998, 0.15999999999999992, 0.010000000000000018],
///                          [0.16000000000000028, 0.0899999999999999, 0.009999999999999929]]);
/// ```
pub fn mse(predictions: &Tensor, targets: &Tensor) -> Tensor {
    (predictions - targets).square()
}

pub fn d_mse(predictions: &Tensor, targets: &Tensor) -> Tensor {
    predictions - targets * 2.0 / predictions.rows as f64
}
