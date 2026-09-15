use crate::{optim::Optimizer, Tensor};

/// Adam, matching `torch.optim.Adam`.
///
/// [`Adam::weight_decay`] adds an L2 penalty to the gradient, while [`Adam::decoupled_weight_decay`] shrinks the weights directly, which is the AdamW formulation.
///
/// ```
/// # use engram::{optim::{Adam, Optimizer}, Tensor};
/// let w = Tensor::zeros([1]).set_requires_grad(true);
/// let mut opt = Adam::new(vec![w.clone()], 0.1);
/// for _ in 0..300 {
///     opt.zero_grad();
///     (&w - 3.0).square().sum().backward();
///     opt.step();
/// }
/// assert!((w.item() - 3.0).abs() < 0.1);
/// ```
#[derive(Debug)]
pub struct Adam {
    parameters: Vec<Tensor>,
    learning_rate: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    decoupled: bool,
    states: Vec<Option<State>>,
}

#[derive(Debug)]
struct State {
    step: i32,
    mean: Vec<f32>,
    variance: Vec<f32>,
}

impl Adam {
    /// Creates an optimizer with betas of `(0.9, 0.999)`, an epsilon of `1e-8` and no weight decay.
    pub fn new(parameters: Vec<Tensor>, learning_rate: f32) -> Adam {
        let states = parameters.iter().map(|_| None).collect();
        Adam {
            parameters,
            learning_rate,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled: false,
            states,
        }
    }

    /// Sets the decay rates of the running gradient mean and variance.
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Adam {
        self.betas = (beta1, beta2);
        self
    }

    /// Sets the term added to the denominator for numerical stability.
    pub fn eps(mut self, eps: f32) -> Adam {
        self.eps = eps;
        self
    }

    /// Adds `weight_decay * parameter` to each gradient, which is an L2 penalty.
    pub fn weight_decay(mut self, weight_decay: f32) -> Adam {
        self.weight_decay = weight_decay;
        self.decoupled = false;
        self
    }

    /// Multiplies weights by `1 - learning_rate * weight_decay` before each update, as in AdamW.
    pub fn decoupled_weight_decay(mut self, weight_decay: f32) -> Adam {
        self.weight_decay = weight_decay;
        self.decoupled = true;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        let (lr, (beta1, beta2), eps) = (self.learning_rate, self.betas, self.eps);
        let (decay, decoupled) = (self.weight_decay, self.decoupled);
        for (parameter, state) in self.parameters.iter().zip(&mut self.states) {
            let Some(grad) = parameter.grad() else {
                continue;
            };
            let grad = grad.data();
            parameter.update(|weights| {
                let state = state.get_or_insert_with(|| State {
                    step: 0,
                    mean: vec![0.0; weights.len()],
                    variance: vec![0.0; weights.len()],
                });
                state.step += 1;
                let mean_correction = 1.0 - beta1.powi(state.step);
                let variance_correction = 1.0 - beta2.powi(state.step);
                let moments = state.mean.iter_mut().zip(&mut state.variance);
                for ((w, &g), (m, v)) in weights.iter_mut().zip(grad.iter()).zip(moments) {
                    let g = if decoupled {
                        *w *= 1.0 - lr * decay;
                        g
                    } else {
                        g + decay * *w
                    };
                    *m = beta1 * *m + (1.0 - beta1) * g;
                    *v = beta2 * *v + (1.0 - beta2) * g * g;
                    let m_hat = *m / mean_correction;
                    let v_hat = *v / variance_correction;
                    *w -= lr * m_hat / (v_hat.sqrt() + eps);
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

    #[test]
    fn first_step_moves_by_learning_rate() {
        let w = tensor![1, -1].set_requires_grad(true);
        let mut opt = Adam::new(vec![w.clone()], 0.01);
        (&w * &tensor![5, -0.1]).sum().backward();
        opt.step();
        assert!(w.allclose(&tensor![0.99, -0.99], 1e-5));
    }

    #[test]
    fn weight_decay_variants_differ() {
        let coupled = tensor![1].set_requires_grad(true);
        let decoupled = tensor![1].set_requires_grad(true);
        let mut a = Adam::new(vec![coupled.clone()], 0.1).weight_decay(0.5);
        let mut b = Adam::new(vec![decoupled.clone()], 0.1).decoupled_weight_decay(0.5);
        (&coupled * 0.0).sum().backward();
        (&decoupled * 0.0).sum().backward();
        a.step();
        b.step();
        assert!(coupled.allclose(&tensor![0.9], 1e-5));
        assert!(decoupled.allclose(&tensor![0.95], 1e-5));
    }
}
