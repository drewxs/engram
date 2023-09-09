//! Adaptive Gradient (Adagrad):
//!
//! Adapts the learning rate based on the history of gradients.
//! Divides the learning rate by a running average of the magnitude of the gradients.
//! This allows the learning rate to decrease for parameters that have consistently large gradients
//! and increase for parameters that have consistently small gradients.
//! Includes an option to apply weight decay regularization to the gradients.

use crate::{Optimize, Tensor};

#[derive(Clone, Debug)]
pub struct Adagrad {
    learning_rate: f64,
    accumulators: Tensor,
    weight_decay: Option<f64>,
    epsilon: Option<f64>,
}

impl Adagrad {
    pub fn new(
        learning_rate: f64,
        shape: (usize, usize),
        weight_decay: Option<f64>,
        epsilon: Option<f64>,
    ) -> Adagrad {
        Adagrad {
            learning_rate,
            epsilon,
            accumulators: Tensor::zeros(shape.0, shape.1),
            weight_decay,
        }
    }
}

impl Optimize for Adagrad {
    fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor) {
        if let Some(wd) = self.weight_decay {
            gradients.sub_assign(&weights.mul_scalar(wd));
        }

        self.accumulators.add_assign(&gradients.square());
        self.accumulators
            .add_scalar_assign(self.epsilon.unwrap_or(1e-8));

        weights.sub_assign(
            &gradients
                .div(&(self.accumulators.mapv(&f64::sqrt)))
                .mul_scalar(self.learning_rate),
        );
    }
}
