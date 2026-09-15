//! Pooling over images.

use std::ops::Range;

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
    let (plane, width) = (window.input_area(), window.input.width);
    let len = output_len(input, &window);
    let mut out = storage::with_capacity(len);
    let mut argmax = Vec::with_capacity(len);
    for (p, channel) in input.data().chunks_exact(plane.max(1)).enumerate() {
        for_each_window(&window, |rows, cols| {
            let first = rows.start * width + cols.start;
            let mut best = (first, channel[first]);
            for iy in rows {
                let start = iy * width + cols.start;
                for (offset, &value) in channel[start..iy * width + cols.end].iter().enumerate() {
                    if value > best.1 {
                        best = (start + offset, value);
                    }
                }
            }
            out.push(best.1);
            argmax.push(p * plane + best.0);
        });
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
    let (plane, area, width) = (
        window.input_area(),
        window.output_area(),
        window.input.width,
    );
    let scale = 1.0 / (window.kernel.height * window.kernel.width) as f32;
    let mut out = storage::with_capacity(output_len(input, &window));
    for channel in input.data().chunks_exact(plane.max(1)) {
        for_each_window(&window, |rows, cols| {
            let sum: f32 = rows
                .map(|iy| {
                    channel[iy * width + cols.start..iy * width + cols.end]
                        .iter()
                        .sum::<f32>()
                })
                .sum();
            out.push(sum * scale);
        });
    }

    let shape = input.shape().to_vec();
    pooled(input, out, &window).record(&[input], move |grad| {
        let mut dx = storage::zeros(shape.iter().product());
        let g = grad.data();
        for (dx, g) in dx
            .chunks_exact_mut(plane.max(1))
            .zip(g.chunks_exact(area.max(1)))
        {
            let mut grads = g.iter();
            for_each_window(&window, |rows, cols| {
                let g = grads.next().unwrap() * scale;
                for iy in rows {
                    for d in &mut dx[iy * width + cols.start..iy * width + cols.end] {
                        *d += g;
                    }
                }
            });
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

/// Calls `f` with the input rows and columns under each window in row-major output order, leaving out padding.
fn for_each_window(window: &Window, mut f: impl FnMut(Range<usize>, Range<usize>)) {
    // Padding is at most half the kernel, so `start + kernel - padding` never underflows.
    let span = |position: usize, stride: usize, padding: usize, kernel: usize, input: usize| {
        let start = position * stride;
        start.saturating_sub(padding)..(start + kernel - padding).min(input)
    };
    let Window {
        input,
        kernel,
        output,
        stride,
        padding,
        ..
    } = *window;
    for oy in 0..output.height {
        let rows = span(
            oy,
            stride.height,
            padding.height,
            kernel.height,
            input.height,
        );
        for ox in 0..output.width {
            let cols = span(ox, stride.width, padding.width, kernel.width, input.width);
            f(rows.clone(), cols);
        }
    }
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
