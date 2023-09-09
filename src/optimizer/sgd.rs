use crate::{Optimize, Tensor};

/// Stochastic Gradient Descent (SGD).
///
/// A basic optimizer that updates the weights based on the gradients of the loss function
/// with respect to the weights multiplied by a learning rate.
#[derive(Clone, Debug)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    /// Creates a new SGD optimizer with the specified parameters.
    pub fn new(learning_rate: f64) -> SGD {
        SGD { learning_rate }
    }
}

impl Optimize for SGD {
    fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor) {
        weights.sub_assign(&gradients.mul_scalar(self.learning_rate));
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_sgd() {
        let mut sgd = SGD::new(0.1);
        let mut weights = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut gradients = weights.gradient(&Activation::ReLU);

        sgd.step(&mut weights, &mut gradients);

        assert_eq!(weights, tensor![[0.9, 1.9, 2.9], [3.9, 4.9, 5.9]]);
    }
}
