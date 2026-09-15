//! Normalization functions.

use crate::{tensor::storage, Tensor};

/// Layer normalization over the trailing `normalized_shape` dimensions, like `torch.nn.functional.layer_norm`.
///
/// Each group of elements spanning those dimensions is shifted to zero mean and scaled to unit variance, then multiplied by `weight` and offset by `bias` when they are given.
///
/// ```
/// # use engram::{nn::functional::layer_norm, tensor};
/// let y = layer_norm(&tensor![[1, 2, 3], [2, 4, 6]], &[3], None, None, 1e-5);
/// assert!(y.allclose(&tensor![[-1.2247, 0, 1.2247], [-1.2247, 0, 1.2247]], 1e-3));
/// ```
///
/// # Panics
///
/// Panics if the input does not end with `normalized_shape`, or `weight` or `bias` has a different shape.
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Tensor {
    let rank = normalized_shape.len();
    assert!(
        rank <= input.ndim() && input.shape()[input.ndim() - rank..] == *normalized_shape,
        "layer_norm: input of shape {:?} does not end with {normalized_shape:?}",
        input.shape()
    );
    for parameter in weight.into_iter().chain(bias) {
        assert_eq!(
            parameter.shape(),
            normalized_shape,
            "layer_norm: weight and bias must have shape {normalized_shape:?}"
        );
    }

    let size = normalized_shape.iter().product::<usize>().max(1);
    let n = size as f32;
    let (x, w, b) = (
        input.data(),
        weight.map(Tensor::data),
        bias.map(Tensor::data),
    );
    let mut normalized = storage::with_capacity(x.len());
    let mut output = storage::with_capacity(x.len());
    let mut inv_std = Vec::with_capacity(x.len() / size);
    for row in x.chunks_exact(size) {
        let mean = row.iter().sum::<f32>() / n;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv = (variance + eps).sqrt().recip();
        inv_std.push(inv);
        for (k, &v) in row.iter().enumerate() {
            let x_hat = (v - mean) * inv;
            normalized.push(x_hat);
            output
                .push(x_hat * w.as_ref().map_or(1.0, |w| w[k]) + b.as_ref().map_or(0.0, |b| b[k]));
        }
    }

    let normalized = Tensor::from_vec(normalized, input.shape());
    let (x, w, b) = (input.clone(), weight.cloned(), bias.cloned());
    let mut inputs = vec![input];
    inputs.extend(weight);
    inputs.extend(bias);
    Tensor::from_vec(output, input.shape()).record(&inputs, move |grad| {
        let (g, x_hat) = (grad.data(), normalized.data());
        let w_data = w.as_ref().map(Tensor::data);
        let scaled = |k: usize, g: f32| w_data.as_ref().map_or(g, |w| g * w[k]);
        let mut dx = storage::zeros(g.len());
        let mut dw = vec![0.0; size];
        let mut db = vec![0.0; size];
        let rows = g.chunks_exact(size).zip(x_hat.chunks_exact(size));
        for (r, ((g, x_hat), dx)) in rows.zip(dx.chunks_exact_mut(size)).enumerate() {
            let (mut sum, mut dot) = (0.0, 0.0);
            for (k, (&g, &x_hat)) in g.iter().zip(x_hat).enumerate() {
                let s = scaled(k, g);
                sum += s;
                dot += s * x_hat;
                dw[k] += g * x_hat;
                db[k] += g;
            }
            let scale = inv_std[r] / n;
            for (k, ((d, &g), &x_hat)) in dx.iter_mut().zip(g).zip(x_hat).enumerate() {
                *d = scale * (n * scaled(k, g) - sum - x_hat * dot);
            }
        }

        let mut grads = vec![x.requires_grad().then(|| Tensor::from_vec(dx, x.shape()))];
        grads.extend(
            w.as_ref()
                .map(|w| w.requires_grad().then(|| Tensor::from_vec(dw, w.shape()))),
        );
        grads.extend(
            b.as_ref()
                .map(|b| b.requires_grad().then(|| Tensor::from_vec(db, b.shape()))),
        );
        grads
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::testing::assert_gradients;

    #[test]
    fn normalizes_each_group() {
        let x = Tensor::randn([2, 3, 4]) * 5.0 + 3.0;
        let y = layer_norm(&x, &[3, 4], None, None, 1e-5);
        let flat = y.reshape([2, 12]);
        assert!(flat.mean_dim(1, false).abs().max().item() < 1e-5);
        assert!((flat.square().mean_dim(1, false) - 1.0).abs().max().item() < 1e-4);

        let (w, b) = (Tensor::randn([4]), Tensor::randn([4]));
        let affine = layer_norm(&x, &[4], Some(&w), Some(&b), 1e-5);
        let expected = layer_norm(&x, &[4], None, None, 1e-5) * &w + &b;
        assert!(affine.allclose(&expected, 1e-5));
    }

    #[test]
    fn gradients_of_layer_norm() {
        let x = Tensor::randn([2, 3, 4]);
        let (w, b) = (Tensor::randn([4]), Tensor::randn([4]));
        assert_gradients([&x, &w, &b], |t| {
            layer_norm(&t[0], &[4], Some(&t[1]), Some(&t[2]), 1e-5)
        });
        assert_gradients([&x], |t| layer_norm(&t[0], &[3, 4], None, None, 1e-5));
    }

    #[test]
    #[should_panic(expected = "does not end with [3]")]
    fn rejects_mismatched_shapes() {
        layer_norm(&Tensor::zeros([2, 4]), &[3], None, None, 1e-5);
    }
}
