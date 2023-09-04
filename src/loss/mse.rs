use crate::Tensor;

/// Mean squared error loss.
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> f64 {
    predictions.sub(targets).pow(2.0).mean()
}
