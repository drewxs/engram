use crate::tensor::Tensor;

/// Mean squared error loss.
pub fn mean_squared_error(predictions: &mut Tensor, targets: &Tensor) -> f64 {
    let diff = predictions.sub(targets);
    diff.clone().pow(2.0).mean()
}

/// Binary cross entropy loss.
pub fn binary_cross_entropy(predictions: &mut Tensor, targets: &Tensor) -> f64 {
    let mut loss = targets.clone();
    loss.mul_assign(&predictions.ln());
    loss.sum()
}
