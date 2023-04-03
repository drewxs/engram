use crate::tensor::Tensor;

#[derive(Debug)]
pub enum Optimizer {
    SGD {
        learning_rate: f64,
    },
    Adagrad {
        learning_rate: f64,
        epsilon: f64,
        accumulators: Tensor,
        weight_decay: Option<f64>,
    },
}

impl Optimizer {
    pub fn step(&mut self, weights: &mut Tensor, gradients: &mut Tensor) {
        match self {
            Optimizer::SGD { learning_rate } => {
                weights.sub_assign(&gradients.mul_scalar(*learning_rate));
            }
            Optimizer::Adagrad {
                learning_rate,
                epsilon,
                accumulators,
                weight_decay,
            } => {
                let sq_gradients = gradients.pow(2.0);

                if let Some(wd) = weight_decay {
                    gradients.sub_assign(&weights.mul_scalar(*wd));
                }

                let mut new_accumulators = sq_gradients.clone();
                new_accumulators.add_assign(&accumulators);

                let mut old_accumulators = std::mem::replace(accumulators, new_accumulators);

                weights.sub_assign(
                    &gradients
                        .div(&old_accumulators.add_scalar(*epsilon).map(&f64::sqrt))
                        .mul_scalar(*learning_rate),
                );
            }
        }
    }
}
