//! Weight initialization, like `torch.nn.init`. Every function overwrites a tensor in place.

use std::f32::consts::SQRT_2;

use crate::{random, Tensor};

/// Fills `tensor` with samples from `U(low, high)`.
pub fn uniform(tensor: &Tensor, low: f32, high: f32) {
    tensor.update(|data| {
        random::with_rng(|rng| data.iter_mut().for_each(|x| *x = rng.uniform(low, high)));
    });
}

/// Fills `tensor` with samples from a normal distribution.
pub fn normal(tensor: &Tensor, mean: f32, std: f32) {
    tensor.update(|data| {
        random::with_rng(|rng| data.iter_mut().for_each(|x| *x = rng.normal(mean, std)));
    });
}

/// Fills `tensor` with `value`.
pub fn constant(tensor: &Tensor, value: f32) {
    tensor.update(|data| data.fill(value));
}

/// Fills `tensor` with zeros.
pub fn zeros(tensor: &Tensor) {
    constant(tensor, 0.0);
}

/// Fills `tensor` with ones.
pub fn ones(tensor: &Tensor) {
    constant(tensor, 1.0);
}

/// Glorot uniform initialization from `U(-a, a)` with `a = gain * sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_uniform(tensor: &Tensor, gain: f32) {
    let (fan_in, fan_out) = fans(tensor);
    let a = gain * (6.0 / (fan_in + fan_out) as f32).sqrt();
    uniform(tensor, -a, a);
}

/// Glorot normal initialization with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`.
pub fn xavier_normal(tensor: &Tensor, gain: f32) {
    let (fan_in, fan_out) = fans(tensor);
    normal(tensor, 0.0, gain * (2.0 / (fan_in + fan_out) as f32).sqrt());
}

/// He uniform initialization for ReLU networks, from `U(-a, a)` with `a = sqrt(6 / fan_in)`.
pub fn kaiming_uniform(tensor: &Tensor) {
    let (fan_in, _) = fans(tensor);
    let a = SQRT_2 * (3.0 / fan_in as f32).sqrt();
    uniform(tensor, -a, a);
}

/// He normal initialization for ReLU networks, with standard deviation `sqrt(2 / fan_in)`.
pub fn kaiming_normal(tensor: &Tensor) {
    let (fan_in, _) = fans(tensor);
    normal(tensor, 0.0, SQRT_2 / (fan_in as f32).sqrt());
}

/// Returns `(fan_in, fan_out)` for a weight shaped `[out, in, ...kernel]`.
fn fans(tensor: &Tensor) -> (usize, usize) {
    assert!(
        tensor.ndim() >= 2,
        "fan computation needs at least 2 dimensions, got shape {:?}",
        tensor.shape()
    );
    let receptive: usize = tensor.shape()[2..].iter().product();
    (tensor.shape()[1] * receptive, tensor.shape()[0] * receptive)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn std(t: &Tensor) -> f32 {
        (t - t.mean()).square().mean().sqrt().item()
    }

    #[test]
    fn scales_with_fan_sizes() {
        let w = Tensor::zeros([200, 300]);
        xavier_uniform(&w, 1.0);
        let a = (6.0f32 / 500.0).sqrt();
        assert!(w.abs().max().item() <= a);
        assert!((std(&w) - a / 3f32.sqrt()).abs() < 0.005);

        kaiming_normal(&w);
        assert!((std(&w) - (2.0f32 / 300.0).sqrt()).abs() < 0.005);

        let conv = Tensor::zeros([16, 8, 3, 3]);
        kaiming_uniform(&conv);
        assert!(conv.abs().max().item() <= (6.0f32 / 72.0).sqrt());
    }

    #[test]
    fn fills_constants() {
        let t = Tensor::zeros([3]);
        constant(&t, 2.5);
        assert_eq!(t.to_vec(), vec![2.5; 3]);
        zeros(&t);
        assert_eq!(t.to_vec(), vec![0.0; 3]);
    }
}
