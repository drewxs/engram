use crate::{optim::Optimizer, Tensor};

/// Stochastic gradient descent with optional momentum, Nesterov momentum and weight decay, matching `torch.optim.SGD`.
///
/// ```
/// # use engram::{optim::{Optimizer, Sgd}, tensor};
/// let w = tensor![1, 2].set_requires_grad(true);
/// let mut opt = Sgd::new(vec![w.clone()], 0.1);
/// (&w * &w).sum().backward();
/// opt.step();
/// assert_eq!(w, tensor![0.8, 1.6]);
/// ```
#[derive(Debug)]
pub struct Sgd {
    parameters: Vec<Tensor>,
    learning_rate: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    velocities: Vec<Option<Vec<f32>>>,
}

impl Sgd {
    /// Creates an optimizer without momentum or weight decay.
    pub fn new(parameters: Vec<Tensor>, learning_rate: f32) -> Sgd {
        let velocities = vec![None; parameters.len()];
        Sgd {
            parameters,
            learning_rate,
            momentum: 0.0,
            nesterov: false,
            weight_decay: 0.0,
            velocities,
        }
    }

    /// Sets the momentum factor.
    pub fn momentum(mut self, momentum: f32) -> Sgd {
        self.momentum = momentum;
        self
    }

    /// Uses Nesterov momentum, which evaluates the update at the look-ahead position.
    pub fn nesterov(mut self, nesterov: bool) -> Sgd {
        self.nesterov = nesterov;
        self
    }

    /// Adds `weight_decay * parameter` to each gradient, which is an L2 penalty.
    pub fn weight_decay(mut self, weight_decay: f32) -> Sgd {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self) {
        let (lr, momentum, nesterov, decay) = (
            self.learning_rate,
            self.momentum,
            self.nesterov,
            self.weight_decay,
        );
        for (parameter, velocity) in self.parameters.iter().zip(&mut self.velocities) {
            let Some(grad) = parameter.grad() else {
                continue;
            };
            let grad = grad.data();
            parameter.update(|weights| {
                if momentum == 0.0 {
                    for (w, &g) in weights.iter_mut().zip(grad.iter()) {
                        *w -= lr * (g + decay * *w);
                    }
                    return;
                }
                let velocity = velocity.get_or_insert_with(|| vec![0.0; weights.len()]);
                for ((w, &g), v) in weights.iter_mut().zip(grad.iter()).zip(velocity) {
                    let g = g + decay * *w;
                    *v = momentum * *v + g;
                    *w -= lr * if nesterov { g + momentum * *v } else { *v };
                }
            });
        }
    }

    fn parameters(&self) -> &[Tensor] {
        &self.parameters
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    fn step_with_grad(opt: &mut Sgd, grad: f32) {
        let w = &opt.parameters()[0];
        opt.zero_grad();
        (w * grad).sum().backward();
        opt.step();
    }

    #[test]
    fn applies_momentum_and_weight_decay() {
        let w = tensor![1].set_requires_grad(true);
        let mut opt = Sgd::new(vec![w.clone()], 0.1).momentum(0.9);
        step_with_grad(&mut opt, 1.0);
        assert!(w.allclose(&tensor![0.9], 1e-6));
        step_with_grad(&mut opt, 1.0);
        assert!(w.allclose(&tensor![0.71], 1e-6));

        let w = tensor![2].set_requires_grad(true);
        let mut opt = Sgd::new(vec![w.clone()], 0.1).weight_decay(0.5);
        step_with_grad(&mut opt, 1.0);
        assert!(w.allclose(&tensor![1.8], 1e-6));
    }

    #[test]
    fn applies_nesterov_momentum() {
        let w = tensor![1].set_requires_grad(true);
        let mut opt = Sgd::new(vec![w.clone()], 0.1).momentum(0.9).nesterov(true);
        step_with_grad(&mut opt, 1.0);
        assert!(w.allclose(&tensor![0.81], 1e-6));
    }

    #[test]
    fn skips_parameters_without_gradients() {
        let (a, b) = (
            tensor![1].set_requires_grad(true),
            tensor![1].set_requires_grad(true),
        );
        let mut opt = Sgd::new(vec![a.clone(), b.clone()], 0.5);
        (&a * 2.0).sum().backward();
        opt.step();
        assert_eq!((a.item(), b.item()), (0.0, 1.0));
    }
}
