//! Training utilities, like `torch.nn.utils`.

use crate::Tensor;

/// Scales gradients in place so that their combined L2 norm is at most `max_norm`, and returns the norm before scaling.
///
/// ```
/// # use engram::{nn::utils::clip_grad_norm, tensor};
/// let (a, b) = (tensor![1].set_requires_grad(true), tensor![1].set_requires_grad(true));
/// ((&a * 3.0) + (&b * 4.0)).sum().backward();
/// assert_eq!(clip_grad_norm(&[a.clone(), b.clone()], 1.0), 5.0);
/// assert!(a.grad().unwrap().allclose(&tensor![0.6], 1e-5));
/// ```
pub fn clip_grad_norm(parameters: &[Tensor], max_norm: f32) -> f32 {
    let grads: Vec<Tensor> = parameters.iter().filter_map(Tensor::grad).collect();
    let squared: f64 = grads
        .iter()
        .map(|g| g.data().iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
        .sum();
    let norm = squared.sqrt() as f32;
    let scale = max_norm / (norm + 1e-6);
    if scale < 1.0 {
        for grad in &grads {
            grad.update(|data| data.iter_mut().for_each(|x| *x *= scale));
        }
    }
    norm
}
