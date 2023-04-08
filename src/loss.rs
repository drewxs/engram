use crate::tensor::Tensor;

/// Mean squared error loss.
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> f64 {
    predictions.sub(targets).pow(2.0).mean()
}

/// Binary cross entropy loss.
pub fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> f64 {
    targets.mul(&predictions.ln()).sum()
}
