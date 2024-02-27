use crate::{Optimize, Tensor};

/// Adaptive Gradient (Adagrad):
///
/// Adapts the learning rate based on the history of gradients.
/// Divides the learning rate by a running average of the magnitude of the gradients.
/// This allows the learning rate to decrease for parameters that have consistently large gradients
/// and increase for parameters that have consistently small gradients.
/// Includes an option to apply weight decay regularization to the gradients.
#[derive(Clone, Debug)]
pub struct Adagrad {
    learning_rate: f64,
    accumulators: Tensor,
    weight_decay: Option<f64>,
    epsilon: Option<f64>,
}

impl Adagrad {
    /// Creates a new Adagrad optimizer with the specified parameters.
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
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) {
        let mut gradients = gradients.clone();
        if let Some(wd) = self.weight_decay {
            gradients -= &*weights * wd;
        }

        self.accumulators += gradients.square() + self.epsilon.unwrap_or(1e-8);

        *weights -= gradients.div(&(self.accumulators.mapv(&f64::sqrt))) * (self.learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_adagrad() {
        let mut adagrad = Adagrad::new(0.1, (2, 3), None, Some(1e-8));
        let mut weights = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut gradients = weights.gradient(&Activation::ReLU);

        adagrad.step(&mut weights, &mut gradients);

        assert_eq!(
            weights,
            tensor![
                [0.9000000005, 1.9000000005, 2.9000000005],
                [3.9000000005, 4.9000000005, 5.9000000005]
            ]
        );
    }
}
