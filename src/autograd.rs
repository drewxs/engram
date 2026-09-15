//! Reverse-mode automatic differentiation.
//!
//! Tensor operations record a graph node on their output whenever an input requires a gradient. [`Tensor::backward`] walks those nodes from the output back to the leaves and accumulates gradients on every leaf that requires one.

use std::{
    cell::Cell,
    collections::{hash_map::Entry, HashMap, HashSet},
};

use crate::{tensor::add_same_shape, Tensor};

pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>;

/// The inputs of a recorded operation and the function mapping its output gradient onto them.
pub(crate) struct Node {
    pub(crate) inputs: Vec<Tensor>,
    pub(crate) backward: BackwardFn,
}

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether operations on the current thread record gradients.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Runs `f` without recording gradients, like `torch.no_grad()`.
///
/// ```
/// # use engram::{no_grad, Tensor};
/// let w = Tensor::ones([3]).set_requires_grad(true);
/// let y = no_grad(|| &w * 2.0);
/// assert!(!y.requires_grad());
/// ```
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    let _guard = GradModeGuard::set(false);
    f()
}

/// Restores the previous grad mode when dropped, even if the guarded code panics.
struct GradModeGuard(bool);

impl GradModeGuard {
    fn set(enabled: bool) -> Self {
        GradModeGuard(GRAD_ENABLED.with(|mode| mode.replace(enabled)))
    }
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|mode| mode.set(self.0));
    }
}

impl Tensor {
    /// Computes the gradient of this single-element tensor with respect to every leaf that requires a gradient.
    ///
    /// Gradients accumulate across calls, so clear them between optimization steps.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has more than one element or does not require a gradient.
    pub fn backward(&self) {
        assert_eq!(
            self.numel(),
            1,
            "backward() expects a single-element tensor, got shape {:?}. Use backward_with_grad for other shapes",
            self.shape()
        );
        self.backward_with_grad(&self.ones_like());
    }

    /// Backpropagates `grad` as the gradient of this tensor.
    ///
    /// # Panics
    ///
    /// Panics if `grad` has a different shape or the tensor does not require a gradient.
    pub fn backward_with_grad(&self, grad: &Tensor) {
        assert!(
            self.requires_grad(),
            "backward() called on a tensor that does not require a gradient"
        );
        assert_eq!(
            self.shape(),
            grad.shape(),
            "backward_with_grad expects a gradient shaped like the tensor"
        );
        let _guard = GradModeGuard::set(false);

        let mut grads = HashMap::from([(self.id(), grad.detach())]);
        for tensor in topological_order(self).iter().rev() {
            let Some(grad) = grads.remove(&tensor.id()) else {
                continue;
            };
            let Some(node) = tensor.node() else {
                tensor.accumulate_grad(grad);
                continue;
            };
            for (input, input_grad) in node.inputs.iter().zip((node.backward)(&grad)) {
                let Some(input_grad) = input_grad.filter(|_| input.requires_grad()) else {
                    continue;
                };
                assert_eq!(
                    input.shape(),
                    input_grad.shape(),
                    "backward produced a gradient with the wrong shape"
                );
                match grads.entry(input.id()) {
                    Entry::Occupied(mut entry) => {
                        let sum = add_same_shape(entry.get(), &input_grad);
                        entry.insert(sum);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(input_grad);
                    }
                }
            }
        }
    }
}

/// Returns every tensor reachable from `root` that requires a gradient, with inputs ordered before the tensors computed from them.
fn topological_order(root: &Tensor) -> Vec<Tensor> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![(root.clone(), false)];
    while let Some((tensor, expanded)) = stack.pop() {
        if expanded {
            order.push(tensor);
            continue;
        }
        if !visited.insert(tensor.id()) {
            continue;
        }
        if let Some(node) = tensor.node() {
            let inputs = node
                .inputs
                .iter()
                .filter(|input| input.requires_grad() && !visited.contains(&input.id()))
                .cloned()
                .collect::<Vec<_>>();
            stack.push((tensor, true));
            stack.extend(inputs.into_iter().map(|input| (input, false)));
        } else {
            stack.push((tensor, true));
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    fn scale(x: &Tensor, k: f32) -> Tensor {
        let data = x.data().iter().map(|v| v * k).collect();
        Tensor::from_vec(data, x.shape()).record(&[x], move |g| vec![Some(scale(g, k))])
    }

    fn add(a: &Tensor, b: &Tensor) -> Tensor {
        add_same_shape(a, b).record(&[a, b], |g| vec![Some(g.clone()), Some(g.clone())])
    }

    #[test]
    fn chains_gradients() {
        let x = tensor![2].set_requires_grad(true);
        scale(&scale(&x, 3.0), 4.0).backward();
        assert_eq!(x.grad().unwrap(), tensor![12]);
    }

    #[test]
    fn sums_gradients_over_shared_inputs() {
        let x = tensor![1, 2].set_requires_grad(true);
        let y = add(&scale(&x, 2.0), &scale(&x, 3.0));
        y.backward_with_grad(&tensor![1, 1]);
        assert_eq!(x.grad().unwrap(), tensor![5, 5]);
    }

    #[test]
    fn accumulates_across_backward_calls() {
        let x = tensor![1].set_requires_grad(true);
        scale(&x, 2.0).backward();
        scale(&x, 2.0).backward();
        assert_eq!(x.grad().unwrap(), tensor![4]);
        x.zero_grad();
        assert!(x.grad().is_none());
    }

    #[test]
    fn skips_tensors_that_do_not_require_grad() {
        let x = tensor![1].set_requires_grad(true);
        let c = tensor![3];
        let y = add(&x, &c);
        assert!(y.requires_grad() && !y.is_leaf());
        y.backward();
        assert!(c.grad().is_none());
        assert_eq!(x.grad().unwrap(), tensor![1]);
    }

    #[test]
    fn no_grad_disables_recording() {
        let x = tensor![1].set_requires_grad(true);
        let y = no_grad(|| scale(&x, 2.0));
        assert!(!y.requires_grad() && y.is_leaf());
        assert!(is_grad_enabled());
    }

    #[test]
    fn detach_cuts_the_graph() {
        let x = tensor![1].set_requires_grad(true);
        let y = scale(&x, 2.0).detach();
        assert!(!y.requires_grad() && y.is_leaf());
    }

    #[test]
    fn handles_long_graphs_without_recursion() {
        let x = tensor![1].set_requires_grad(true);
        let mut y = x.clone();
        for _ in 0..100_000 {
            y = scale(&y, 1.0);
        }
        y.backward();
        assert_eq!(x.grad().unwrap(), tensor![1]);
        drop(y);
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use crate::{manual_seed, no_grad, Tensor};

    /// Asserts that the gradients of `f` at `inputs` match central finite differences.
    pub(crate) fn assert_gradients<const N: usize>(
        inputs: [&Tensor; N],
        f: impl Fn(&[Tensor]) -> Tensor,
    ) {
        let inputs: Vec<Tensor> = inputs
            .iter()
            .map(|t| t.detach().set_requires_grad(true))
            .collect();
        let output = f(&inputs);
        manual_seed(0);
        let weights = Tensor::randn(output.shape());
        output.backward_with_grad(&weights);

        let objective = |inputs: &[Tensor]| -> f64 {
            let output = no_grad(|| f(inputs)).data();
            let weights = weights.data();
            output
                .iter()
                .zip(weights.iter())
                .map(|(&o, &w)| o as f64 * w as f64)
                .sum()
        };
        let eps = 1e-2;
        for (k, input) in inputs.iter().enumerate() {
            let analytic = input.grad().expect("input received no gradient").to_vec();
            for (i, &analytic) in analytic.iter().enumerate() {
                let original = input.data()[i];
                input.update(|data| data[i] = original + eps);
                let plus = objective(&inputs);
                input.update(|data| data[i] = original - eps);
                let minus = objective(&inputs);
                input.update(|data| data[i] = original);
                let numeric = ((plus - minus) / (2.0 * eps as f64)) as f32;
                assert!(
                    (numeric - analytic).abs() <= 1e-2 * (1.0 + numeric.abs()),
                    "input {k} element {i}: analytic gradient {analytic} but numeric {numeric}"
                );
            }
        }
    }
}
