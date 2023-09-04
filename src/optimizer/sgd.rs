//! Stochastic Gradient Descent (SGD)
//!
//! A basic optimizer that updates the weights based on the gradients of the loss function
//! with respect to the weights multiplied by a learning rate.

use crate::{Optimize, Tensor};

#[derive(Clone, Debug)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> SGD {
        SGD { learning_rate }
    }
}

impl Optimize for SGD {
    fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor) {
        weights.sub_assign(&gradients.mul_scalar(self.learning_rate));
    }
}
