//! Stateless functions behind the layers, like `torch.nn.functional`.

mod loss;

pub use loss::*;

use crate::{
    random,
    tensor::{batched_gemm, storage::CollectBuffer},
    Tensor,
};

/// Applies the affine map `input @ weight.T + bias` over the last dimension.
///
/// `input` has shape `[..., in]`, `weight` has shape `[out, in]` and `bias` has shape `[out]`. Leading dimensions fold into one matrix product, and the transposed weight is never copied.
///
/// ```
/// # use engram::{nn::functional::linear, Tensor};
/// let y = linear(&Tensor::ones([5, 2, 3]), &Tensor::ones([4, 3]), Some(&Tensor::zeros([4])));
/// assert_eq!(y.shape(), &[5, 2, 4]);
/// assert_eq!(y.max().item(), 3.0);
/// ```
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    assert_eq!(
        weight.ndim(),
        2,
        "linear: weight must be 2-d, got shape {:?}",
        weight.shape()
    );
    let (out_features, in_features) = (weight.shape()[0], weight.shape()[1]);
    assert!(
        input.shape().last() == Some(&in_features),
        "linear: input of shape {:?} does not end with {in_features} features",
        input.shape()
    );
    if let Some(bias) = bias {
        assert_eq!(
            bias.shape(),
            &[out_features],
            "linear: bias must have shape [{out_features}]"
        );
    }

    let rows = input.numel() / in_features.max(1);
    let out = batched_gemm(&as_matrix(input, rows, in_features), false, weight, true);
    if let Some(bias) = bias {
        let b = bias.data();
        out.update(|data| {
            for row in data.chunks_exact_mut(out_features.max(1)) {
                row.iter_mut().zip(b.iter()).for_each(|(y, b)| *y += b);
            }
        });
    }
    let mut out_shape = input.shape().to_vec();
    *out_shape.last_mut().unwrap() = out_features;

    let (x, w, b) = (input.clone(), weight.clone(), bias.cloned());
    let mut inputs = vec![input, weight];
    inputs.extend(bias);
    out.reshape(out_shape).record(&inputs, move |g| {
        let g = as_matrix(g, rows, out_features);
        let mut grads = vec![
            x.requires_grad()
                .then(|| batched_gemm(&g, false, &w, false).reshape(x.shape())),
            w.requires_grad()
                .then(|| batched_gemm(&g, true, &as_matrix(&x, rows, in_features), false)),
        ];
        if let Some(b) = &b {
            grads.push(b.requires_grad().then(|| g.sum_dim(0, false)));
        }
        grads
    })
}

/// Views the data of `tensor` as a `rows` by `cols` matrix without tracking gradients.
fn as_matrix(tensor: &Tensor, rows: usize, cols: usize) -> Tensor {
    tensor.detach().reshape([rows, cols])
}

/// Zeroes each element with probability `p` and scales the rest by `1 / (1 - p)`, which keeps the expected value unchanged. Returns the input as is when `training` is false.
///
/// # Panics
///
/// Panics if `p` is outside `[0, 1]`.
pub fn dropout(input: &Tensor, p: f32, training: bool) -> Tensor {
    assert!(
        (0.0..=1.0).contains(&p),
        "dropout probability must be in [0, 1], got {p}"
    );
    if !training || p == 0.0 {
        return input.clone();
    }
    let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
    let mask = random::with_rng(|rng| {
        (0..input.numel())
            .map(|_| if rng.next_f32() < p { 0.0 } else { scale })
            .collect_buffer()
    });
    input * Tensor::from_vec(mask, input.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::testing::assert_gradients;

    #[test]
    fn linear_matches_matmul() {
        let (x, w, b) = (
            Tensor::randn([2, 5, 3]),
            Tensor::randn([4, 3]),
            Tensor::randn([4]),
        );
        let expected = x.matmul(&w.t()) + &b;
        assert!(linear(&x, &w, Some(&b)).allclose(&expected, 1e-5));
        assert!(linear(&x, &w, None).allclose(&x.matmul(&w.t()), 1e-5));
    }

    #[test]
    fn gradients_of_linear() {
        let (x, w, b) = (
            Tensor::randn([2, 5, 3]),
            Tensor::randn([4, 3]),
            Tensor::randn([4]),
        );
        assert_gradients([&x, &w, &b], |t| linear(&t[0], &t[1], Some(&t[2])));
        assert_gradients([&x, &w], |t| linear(&t[0], &t[1], None));
    }

    #[test]
    fn dropout_preserves_expectation() {
        let x = Tensor::ones([100_000]);
        let y = dropout(&x, 0.3, true);
        let kept = y.greater(0.0).mean().item();
        assert!((kept - 0.7).abs() < 0.01, "kept {kept}");
        assert!((y.mean().item() - 1.0).abs() < 0.02);
        assert_eq!(dropout(&x, 0.3, false), x);
        assert_eq!(dropout(&x, 1.0, true).max().item(), 0.0);
    }

    #[test]
    fn gradients_flow_through_dropout_mask() {
        let x = Tensor::ones([10]).set_requires_grad(true);
        let y = dropout(&x, 0.5, true);
        y.sum().backward();
        assert_eq!(x.grad().unwrap(), y.detach());
    }
}
