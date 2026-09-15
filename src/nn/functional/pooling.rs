//! Pooling over images.

use crate::{
    nn::functional::conv::{Size2, Window},
    tensor::storage,
    Tensor,
};

/// 2-d max pooling over a batch of images of shape `[N, C, H, W]`, like `torch.nn.functional.max_pool2d`.
///
/// ```
/// # use engram::{nn::functional::max_pool2d, Tensor};
/// let x = Tensor::arange(0.0, 16.0, 1.0).reshape([1, 1, 4, 4]);
/// assert_eq!(max_pool2d(&x, 2, 2, 0).to_vec(), vec![5.0, 7.0, 13.0, 15.0]);
/// ```
///
/// # Panics
///
/// Panics if the input is not 4-d, the padding exceeds half the kernel, or the kernel does not fit.
pub fn max_pool2d(
    input: &Tensor,
    kernel_size: impl Into<Size2>,
    stride: impl Into<Size2>,
    padding: impl Into<Size2>,
) -> Tensor {
    let window = pooling_window(
        "max_pool2d",
        input,
        kernel_size.into(),
        stride.into(),
        padding.into(),
    );
    let plane = window.input_area();
    let len = output_len(input, &window);
    let mut out = storage::with_capacity(len);
    let mut argmax = Vec::with_capacity(len);
    for (p, channel) in input.data().chunks_exact(plane.max(1)).enumerate() {
        for (oy, ox) in output_positions(&window) {
            let (index, value) = covered(&window, oy, ox)
                .map(|i| (i, channel[i]))
                .reduce(|best, next| if next.1 > best.1 { next } else { best })
                .unwrap();
            out.push(value);
            argmax.push(p * plane + index);
        }
    }

    let shape = input.shape().to_vec();
    pooled(input, out, &window).record(&[input], move |grad| {
        let mut dx = storage::zeros(shape.iter().product());
        for (&index, &g) in argmax.iter().zip(grad.data().iter()) {
            dx[index] += g;
        }
        vec![Some(Tensor::from_vec(dx, shape.clone()))]
    })
}

/// 2-d average pooling over a batch of images of shape `[N, C, H, W]`, like `torch.nn.functional.avg_pool2d`.
///
/// Padded positions count toward the average, matching the PyTorch default.
///
/// ```
/// # use engram::{nn::functional::avg_pool2d, Tensor};
/// let x = Tensor::arange(0.0, 16.0, 1.0).reshape([1, 1, 4, 4]);
/// assert_eq!(avg_pool2d(&x, 2, 2, 0).to_vec(), vec![2.5, 4.5, 10.5, 12.5]);
/// ```
///
/// # Panics
///
/// Panics if the input is not 4-d, the padding exceeds half the kernel, or the kernel does not fit.
pub fn avg_pool2d(
    input: &Tensor,
    kernel_size: impl Into<Size2>,
    stride: impl Into<Size2>,
    padding: impl Into<Size2>,
) -> Tensor {
    let window = pooling_window(
        "avg_pool2d",
        input,
        kernel_size.into(),
        stride.into(),
        padding.into(),
    );
    let (plane, area) = (window.input_area(), window.output_area());
    let scale = 1.0 / (window.kernel.height * window.kernel.width) as f32;
    let mut out = storage::with_capacity(output_len(input, &window));
    for channel in input.data().chunks_exact(plane.max(1)) {
        for (oy, ox) in output_positions(&window) {
            out.push(covered(&window, oy, ox).map(|i| channel[i]).sum::<f32>() * scale);
        }
    }

    let shape = input.shape().to_vec();
    pooled(input, out, &window).record(&[input], move |grad| {
        let mut dx = storage::zeros(shape.iter().product());
        let g = grad.data();
        for (dx, g) in dx
            .chunks_exact_mut(plane.max(1))
            .zip(g.chunks_exact(area.max(1)))
        {
            for ((oy, ox), &g) in output_positions(&window).zip(g) {
                for i in covered(&window, oy, ox) {
                    dx[i] += g * scale;
                }
            }
        }
        vec![Some(Tensor::from_vec(dx, shape.clone()))]
    })
}

fn pooling_window(
    op: &str,
    input: &Tensor,
    kernel: Size2,
    stride: Size2,
    padding: Size2,
) -> Window {
    let &[_, _, height, width] = input.shape() else {
        panic!("{op}: expected input [N, C, H, W], got {:?}", input.shape());
    };
    assert!(
        padding.height <= kernel.height / 2 && padding.width <= kernel.width / 2,
        "{op}: padding must be at most half the kernel size"
    );
    Window::new(
        op,
        Size2 { height, width },
        kernel,
        stride,
        padding,
        Size2::from(1),
    )
}

fn output_positions(window: &Window) -> impl Iterator<Item = (usize, usize)> {
    let Size2 { height, width } = window.output;
    (0..height).flat_map(move |oy| (0..width).map(move |ox| (oy, ox)))
}

/// Returns the flat indices within one input plane that the window at `(oy, ox)` covers, skipping padding.
fn covered(window: &Window, oy: usize, ox: usize) -> impl Iterator<Item = usize> + '_ {
    (0..window.kernel.height)
        .filter_map(move |ky| window.row(oy, ky))
        .flat_map(move |iy| {
            (0..window.kernel.width)
                .filter_map(move |kx| window.col(ox, kx))
                .map(move |ix| iy * window.input.width + ix)
        })
}

fn output_len(input: &Tensor, window: &Window) -> usize {
    input.shape()[0] * input.shape()[1] * window.output_area()
}

fn pooled(input: &Tensor, data: Vec<f32>, window: &Window) -> Tensor {
    let shape = [
        input.shape()[0],
        input.shape()[1],
        window.output.height,
        window.output.width,
    ];
    Tensor::from_vec(data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::testing::assert_gradients;

    #[test]
    fn pools_padded_and_overlapping_windows() {
        let x = Tensor::arange(0.0, 16.0, 1.0).reshape([1, 1, 4, 4]);
        let max = max_pool2d(&x, 3, 1, 1);
        assert_eq!(max.shape(), &[1, 1, 4, 4]);
        assert_eq!(max.to_vec()[..4], [5.0, 6.0, 7.0, 7.0]);
        let avg = avg_pool2d(&x, 3, 2, 1);
        assert_eq!(avg.shape(), &[1, 1, 2, 2]);
        assert_eq!(avg.to_vec()[0], (0.0 + 1.0 + 4.0 + 5.0) / 9.0);
        let rectangular = max_pool2d(&Tensor::zeros([2, 3, 7, 5]), (2, 3), (2, 1), 0);
        assert_eq!(rectangular.shape(), &[2, 3, 3, 3]);
    }

    #[test]
    fn gradients_of_pooling() {
        // Distinct, well separated values keep finite differences from switching the selected maximum.
        let mut values: Vec<f32> = (0..50).map(|i| i as f32).collect();
        crate::random::shuffle(&mut values);
        let separated = Tensor::from_vec(values, [1, 2, 5, 5]);
        assert_gradients([&separated], |t| max_pool2d(&t[0], 3, 2, 1));
        assert_gradients([&Tensor::randn([2, 2, 5, 4])], |t| {
            avg_pool2d(&t[0], (3, 2), 2, 1)
        });
    }

    #[test]
    #[should_panic(expected = "padding must be at most half the kernel size")]
    fn rejects_oversized_padding() {
        max_pool2d(&Tensor::zeros([1, 1, 4, 4]), 2, 2, 2);
    }
}
