//! Mean squared error loss.

use crate::Tensor;

/// Returns mean squared error between the predictions and the targets.
///
/// # Examples
///
/// ```
/// # use engram::{tensor, loss::mean_squared_error};
/// let predictions = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let targets = tensor![[1.2, 2.4, 3.1], [4.4, 4.7, 5.9]];
/// let loss = mean_squared_error(&predictions, &targets);
/// assert_eq!(loss, 0.07833333333333335);
/// ```
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> f64 {
    predictions.sub(targets).square().mean()
}
