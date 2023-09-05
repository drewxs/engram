//! Binary cross entropy loss.

use crate::Tensor;

/// Returns the binary cross entropy between the predictions and the targets.
///
/// # Examples
///
/// ```
/// # use engram::{tensor, loss::binary_cross_entropy};
/// let predictions = tensor![[0.9, 0.1, 0.2], [0.1, 0.9, 0.8]];
/// let targets = tensor![[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
/// let loss = binary_cross_entropy(&predictions, &targets);
/// assert_eq!(loss, 0.14462152754328741);
/// ```
pub fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> f64 {
    assert_eq!(
        predictions.shape(),
        targets.shape(),
        "Shapes of predictions and targets must match."
    );

    let epsilon = 1e-15; // Small constant to avoid log(0)
    let predictions = predictions.clip(epsilon, 1.0 - epsilon);
    let ones = Tensor::ones_like(&predictions);
    let predictions_complement = ones.sub(&predictions);
    let targets_complement = ones.sub(&targets);

    let loss = targets
        .mul(&predictions.ln())
        .add(&targets_complement.mul(&predictions_complement.ln()))
        .mul_scalar(-1.0);
    let mean_loss = loss.mean();

    mean_loss
}
