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

/// Batch normalization over the channel dimension, like `torch.nn.functional.batch_norm`.
///
/// `input` has shape `[N, C, ...]`, and statistics are taken over every dimension except the channels. In training mode each channel is normalized with the mean and variance of the batch, and `running_stats`, a running mean and variance, move toward them by `momentum`. In evaluation mode the running statistics are used instead.
///
/// ```
/// # use engram::{nn::functional::batch_norm, Tensor};
/// let x = Tensor::randn([16, 3, 8, 8]) * 4.0 + 2.0;
/// let y = batch_norm(&x, None, None, None, true, 0.1, 1e-5);
/// let channel_means = y.transpose(0, 1).flatten(1, -1).mean_dim(1, false);
/// assert!(channel_means.abs().max().item() < 1e-4);
/// ```
///
/// # Panics
///
/// Panics if the input has fewer than two dimensions, a parameter or statistic is not shaped `[C]`, training sees only one value per channel, or evaluation mode has no running statistics.
pub fn batch_norm(
    input: &Tensor,
    running_stats: Option<(&Tensor, &Tensor)>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Tensor {
    assert!(
        input.ndim() >= 2,
        "batch_norm: expected input [N, C, ...], got {:?}",
        input.shape()
    );
    let channels = input.shape()[1];
    let block = input.shape()[2..].iter().product::<usize>().max(1);
    let stats = running_stats
        .into_iter()
        .flat_map(|(mean, var)| [mean, var]);
    for tensor in weight.into_iter().chain(bias).chain(stats) {
        assert_eq!(
            tensor.shape(),
            &[channels],
            "batch_norm: parameters and running statistics must have shape [{channels}]"
        );
    }

    let x = input.data();
    let count = input.numel() / channels.max(1);
    let (mean, var) = if training {
        assert!(
            count > 1,
            "batch_norm: training needs more than one value per channel"
        );
        let (mean, var) = channel_moments(&x, channels, block, count);
        if let Some((running_mean, running_var)) = running_stats {
            let unbiased = count as f32 / (count - 1) as f32;
            running_mean.update(|r| {
                r.iter_mut()
                    .zip(&mean)
                    .for_each(|(r, &m)| *r += momentum * (m - *r));
            });
            running_var.update(|r| {
                r.iter_mut()
                    .zip(&var)
                    .for_each(|(r, &v)| *r += momentum * (v * unbiased - *r));
            });
        }
        (mean, var)
    } else {
        let (running_mean, running_var) =
            running_stats.expect("batch_norm: evaluation mode needs running statistics");
        (running_mean.to_vec(), running_var.to_vec())
    };

    let inv_std: Vec<f32> = var.iter().map(|v| (v + eps).sqrt().recip()).collect();
    let (w, b) = (weight.map(Tensor::data), bias.map(Tensor::data));
    let mut normalized = storage::with_capacity(x.len());
    let mut output = storage::with_capacity(x.len());
    for (i, values) in x.chunks_exact(block).enumerate() {
        let c = i % channels;
        let scale = w.as_ref().map_or(1.0, |w| w[c]);
        let shift = b.as_ref().map_or(0.0, |b| b[c]);
        for &v in values {
            let x_hat = (v - mean[c]) * inv_std[c];
            normalized.push(x_hat);
            output.push(x_hat * scale + shift);
        }
    }

    let normalized = Tensor::from_vec(normalized, input.shape());
    let (input_t, weight_t, bias_t) = (input.clone(), weight.cloned(), bias.cloned());
    let mut inputs = vec![input];
    inputs.extend(weight);
    inputs.extend(bias);
    Tensor::from_vec(output, input.shape()).record(&inputs, move |grad| {
        let (g, x_hat) = (grad.data(), normalized.data());
        let w = weight_t.as_ref().map(Tensor::data);
        let (mut dw, mut db) = (vec![0.0; channels], vec![0.0; channels]);
        for (i, (g, x_hat)) in g
            .chunks_exact(block)
            .zip(x_hat.chunks_exact(block))
            .enumerate()
        {
            let c = i % channels;
            for (&g, &x_hat) in g.iter().zip(x_hat) {
                dw[c] += g * x_hat;
                db[c] += g;
            }
        }
        // With batch statistics every output depends on the whole channel, which adds the two mean terms.
        let dx = input_t.requires_grad().then(|| {
            let n = count as f32;
            let mut dx = storage::zeros(g.len());
            let rows = g.chunks_exact(block).zip(x_hat.chunks_exact(block));
            for (i, (dx, (g, x_hat))) in dx.chunks_exact_mut(block).zip(rows).enumerate() {
                let c = i % channels;
                let k = w.as_ref().map_or(1.0, |w| w[c]) * inv_std[c];
                for ((d, &g), &x_hat) in dx.iter_mut().zip(g).zip(x_hat) {
                    *d = if training {
                        k * (g - (db[c] + x_hat * dw[c]) / n)
                    } else {
                        k * g
                    };
                }
            }
            Tensor::from_vec(dx, input_t.shape())
        });

        let mut grads = vec![dx];
        grads.extend(
            weight_t
                .as_ref()
                .map(|w| w.requires_grad().then(|| Tensor::from_vec(dw, w.shape()))),
        );
        grads.extend(
            bias_t
                .as_ref()
                .map(|b| b.requires_grad().then(|| Tensor::from_vec(db, b.shape()))),
        );
        grads
    })
}

/// Returns the mean and population variance of each channel of data laid out as `[N, C, block]`.
fn channel_moments(x: &[f32], channels: usize, block: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sums = vec![0.0f64; channels];
    let mut squares = vec![0.0f64; channels];
    for (i, values) in x.chunks_exact(block).enumerate() {
        let c = i % channels;
        for &v in values {
            sums[c] += v as f64;
            squares[c] += (v as f64).powi(2);
        }
    }
    sums.iter()
        .zip(&squares)
        .map(|(&sum, &square)| {
            let mean = sum / count as f64;
            (
                mean as f32,
                (square / count as f64 - mean * mean).max(0.0) as f32,
            )
        })
        .unzip()
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

    #[test]
    fn batch_norm_normalizes_channels_and_tracks_statistics() {
        let x = Tensor::randn([32, 2, 5]) * 3.0 + 1.0;
        let (running_mean, running_var) = (Tensor::zeros([2]), Tensor::ones([2]));
        let stats = Some((&running_mean, &running_var));
        let y = batch_norm(&x, stats, None, None, true, 0.1, 1e-5);
        let per_channel = y.transpose(0, 1).flatten(1, -1);
        assert!(per_channel.mean_dim(1, false).abs().max().item() < 1e-4);
        assert!(
            (per_channel.square().mean_dim(1, false) - 1.0)
                .abs()
                .max()
                .item()
                < 1e-3
        );

        let batch = x.transpose(0, 1).flatten(1, -1);
        let mean = batch.mean_dim(1, false);
        let unbiased = (&batch - &mean.unsqueeze(1)).square().sum_dim(1, false) / 159.0;
        assert!(running_mean.allclose(&(&mean * 0.1), 1e-4));
        assert!(running_var.allclose(&(unbiased * 0.1 + 0.9), 1e-3));
    }

    #[test]
    fn batch_norm_uses_running_statistics_in_evaluation() {
        let x = crate::tensor![[1, 2], [3, 4]];
        let (mean, var) = (crate::tensor![1, 2], crate::tensor![4, 1]);
        let (weight, bias) = (crate::tensor![2, 1], crate::tensor![0, 1]);
        let y = batch_norm(
            &x,
            Some((&mean, &var)),
            Some(&weight),
            Some(&bias),
            false,
            0.1,
            0.0,
        );
        assert_eq!(y, crate::tensor![[0, 1], [2, 3]]);
    }

    #[test]
    fn gradients_of_batch_norm() {
        let x = Tensor::randn([4, 3, 2, 2]);
        let (w, b) = (Tensor::randn([3]), Tensor::randn([3]));
        assert_gradients([&x, &w, &b], |t| {
            batch_norm(&t[0], None, Some(&t[1]), Some(&t[2]), true, 0.1, 1e-5)
        });
        let (mean, var) = (Tensor::randn([3]), Tensor::rand([3]) + 0.5);
        assert_gradients([&x, &w], |t| {
            batch_norm(
                &t[0],
                Some((&mean, &var)),
                Some(&t[1]),
                None,
                false,
                0.1,
                1e-5,
            )
        });
    }
}
