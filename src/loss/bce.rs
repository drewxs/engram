use crate::Tensor;

/// Binary cross entropy loss.
pub fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> f64 {
    let epsilon = 1e-8; // Small value to avoid taking the logarithm of zero
    let loss = &(predictions.add_scalar(epsilon).ln().mul(&targets).add(
        &targets
            .sub_scalar(1.0)
            .mul(&predictions.sub_scalar(1.0).add_scalar(epsilon).ln()),
    ))
    .mul_scalar(-1.0);
    loss.mean()
}
