//! Loss functions. Each one averages over the batch and returns a scalar.

use crate::{tensor::sigmoid, Tensor};

/// Mean squared error.
///
/// ```
/// # use engram::{nn::functional::mse_loss, tensor};
/// assert_eq!(mse_loss(&tensor![1, 2], &tensor![1, 4]).item(), 2.0);
/// ```
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    expect_same_shape("mse_loss", input, target);
    (input - target).square().mean()
}

/// Mean absolute error.
pub fn l1_loss(input: &Tensor, target: &Tensor) -> Tensor {
    expect_same_shape("l1_loss", input, target);
    (input - target).abs().mean()
}

/// Binary cross entropy between predicted probabilities and targets in `[0, 1]`.
///
/// Log terms are clamped at -100 as in PyTorch, so probabilities of exactly 0 or 1 still give a finite loss. Prefer [`binary_cross_entropy_with_logits`] when the model outputs raw scores.
pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    expect_same_shape("binary_cross_entropy", input, target);
    pointwise_loss(
        input,
        target,
        |p, t| -(t * p.ln().max(-100.0) + (1.0 - t) * (1.0 - p).ln().max(-100.0)),
        |p, t| (p - t) / ((1.0 - p) * p).max(1e-12),
    )
}

/// Binary cross entropy on raw scores, combining the sigmoid and the loss so large logits stay numerically stable.
///
/// ```
/// # use engram::{nn::functional::binary_cross_entropy_with_logits, tensor};
/// let loss = binary_cross_entropy_with_logits(&tensor![0], &tensor![1]);
/// assert!((loss.item() - 2f32.ln()).abs() < 1e-6);
/// ```
pub fn binary_cross_entropy_with_logits(input: &Tensor, target: &Tensor) -> Tensor {
    expect_same_shape("binary_cross_entropy_with_logits", input, target);
    pointwise_loss(
        input,
        target,
        |x, t| x.max(0.0) - x * t + (-x.abs()).exp().ln_1p(),
        |x, t| sigmoid(x) - t,
    )
}

/// Cross entropy between raw scores and class indices, like `torch.nn.functional.cross_entropy`.
///
/// `input` has shape `[N, C]` and `target` holds the class index of each sample with shape `[N]`. The log-softmax is applied internally.
///
/// ```
/// # use engram::{nn::functional::cross_entropy, tensor};
/// let logits = tensor![[4, 0, 0], [0, 0, 4]];
/// assert!(cross_entropy(&logits, &tensor![0, 2]).item() < 0.05);
/// ```
pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    let labels = class_labels("cross_entropy", input, target);
    let classes = input.shape()[1];
    let n = labels.len() as f32;
    let total: f64 = input
        .data()
        .chunks_exact(classes)
        .zip(&labels)
        .map(|(row, &label)| (log_sum_exp(row) - row[label]) as f64)
        .sum();

    let x = input.clone();
    Tensor::scalar((total / n as f64) as f32).record(&[input], move |g| {
        let scale = g.item() / n;
        let mut grad = x.to_vec();
        for (row, &label) in grad.chunks_exact_mut(classes).zip(&labels) {
            let lse = log_sum_exp(row);
            row.iter_mut().for_each(|v| *v = (*v - lse).exp() * scale);
            row[label] -= scale;
        }
        vec![Some(Tensor::from_vec(grad, x.shape()))]
    })
}

/// Negative log likelihood of class indices, given log-probabilities of shape `[N, C]` and targets of shape `[N]`.
pub fn nll_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let labels = class_labels("nll_loss", input, target);
    let classes = input.shape()[1];
    let n = labels.len() as f32;
    let log_probs = input.data();
    let total: f64 = labels
        .iter()
        .enumerate()
        .map(|(i, &label)| -log_probs[i * classes + label] as f64)
        .sum();

    let shape = input.shape().to_vec();
    Tensor::scalar((total / n as f64) as f32).record(&[input], move |g| {
        let scale = -g.item() / n;
        let mut grad = vec![0.0; labels.len() * classes];
        for (i, &label) in labels.iter().enumerate() {
            grad[i * classes + label] = scale;
        }
        vec![Some(Tensor::from_vec(grad, shape.clone()))]
    })
}

/// Averages `loss(input, target)` over all elements, where `grad` is the derivative of each term with respect to its input.
fn pointwise_loss(
    input: &Tensor,
    target: &Tensor,
    loss: impl Fn(f32, f32) -> f32,
    grad: impl Fn(f32, f32) -> f32 + Send + Sync + 'static,
) -> Tensor {
    let n = input.numel() as f32;
    let total: f64 = input
        .data()
        .iter()
        .zip(target.data().iter())
        .map(|(&x, &t)| loss(x, t) as f64)
        .sum();

    let (x, t) = (input.clone(), target.clone());
    Tensor::scalar((total / n as f64) as f32).record(&[input], move |g| {
        let scale = g.item() / n;
        let data = x
            .data()
            .iter()
            .zip(t.data().iter())
            .map(|(&x, &t)| scale * grad(x, t))
            .collect();
        vec![Some(Tensor::from_vec(data, x.shape()))]
    })
}

fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

/// Checks the shapes of a classification loss and converts the target into class indices.
fn class_labels(op: &str, input: &Tensor, target: &Tensor) -> Vec<usize> {
    assert!(
        input.ndim() == 2 && target.ndim() == 1 && target.shape()[0] == input.shape()[0],
        "{op}: expected input [N, C] and target [N], got {:?} and {:?}",
        input.shape(),
        target.shape()
    );
    let classes = input.shape()[1];
    target
        .data()
        .iter()
        .map(|&t| {
            assert!(
                t >= 0.0 && t.fract() == 0.0 && (t as usize) < classes,
                "{op}: target {t} is not a class index below {classes}"
            );
            t as usize
        })
        .collect()
}

fn expect_same_shape(op: &str, input: &Tensor, target: &Tensor) {
    assert_eq!(
        input.shape(),
        target.shape(),
        "{op}: input shape {:?} does not match target shape {:?}",
        input.shape(),
        target.shape()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autograd::testing::assert_gradients, tensor};

    #[test]
    fn regression_losses() {
        let (x, y) = (tensor![1, -2, 3], tensor![0, 0, 3]);
        assert_eq!(mse_loss(&x, &y).item(), 5.0 / 3.0);
        assert_eq!(l1_loss(&x, &y).item(), 1.0);
        let x = Tensor::randn([4, 2]);
        let y = Tensor::randn([4, 2]);
        assert_gradients([&x], |t| mse_loss(&t[0], &y));
        assert_gradients([&(&x + 5.0)], |t| l1_loss(&t[0], &y));
    }

    #[test]
    fn binary_cross_entropy_matches_definition() {
        let p = tensor![0.9, 0.2];
        let t = tensor![1, 0];
        let expected = -(0.9f32.ln() + 0.8f32.ln()) / 2.0;
        assert!((binary_cross_entropy(&p, &t).item() - expected).abs() < 1e-6);
        assert_eq!(binary_cross_entropy(&tensor![0], &tensor![1]).item(), 100.0);

        let logits = Tensor::randn([6]);
        let targets = Tensor::rand([6]);
        let via_sigmoid = binary_cross_entropy(&logits.sigmoid(), &targets);
        let fused = binary_cross_entropy_with_logits(&logits, &targets);
        assert!((via_sigmoid.item() - fused.item()).abs() < 1e-5);
        assert!(binary_cross_entropy_with_logits(&tensor![100], &tensor![0]).item() == 100.0);
    }

    #[test]
    fn gradients_of_binary_losses() {
        let targets = Tensor::rand([6]);
        let probs = Tensor::rand([6]) * 0.8 + 0.1;
        assert_gradients([&probs], |t| binary_cross_entropy(&t[0], &targets));
        assert_gradients([&Tensor::randn([6])], |t| {
            binary_cross_entropy_with_logits(&t[0], &targets)
        });
    }

    #[test]
    fn cross_entropy_matches_log_softmax() {
        let logits = Tensor::randn([5, 4]);
        let target = tensor![0, 3, 1, 1, 2];
        let expected = nll_loss(&logits.log_softmax(1), &target);
        assert!((cross_entropy(&logits, &target).item() - expected.item()).abs() < 1e-5);
        assert_gradients([&logits], |t| cross_entropy(&t[0], &target));
        assert_gradients([&logits.log_softmax(1)], |t| nll_loss(&t[0], &target));
    }

    #[test]
    #[should_panic(expected = "cross_entropy: target 4 is not a class index below 4")]
    fn rejects_out_of_range_classes() {
        cross_entropy(&Tensor::zeros([1, 4]), &tensor![4]);
    }
}
