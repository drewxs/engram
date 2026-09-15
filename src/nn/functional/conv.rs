//! Convolution over images.

use std::ops::Range;

use crate::{
    tensor::{gemm, storage},
    Tensor,
};

/// A height and width. A single number applies to both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size2 {
    /// Number of rows.
    pub height: usize,
    /// Number of columns.
    pub width: usize,
}

impl From<usize> for Size2 {
    fn from(size: usize) -> Size2 {
        Size2 {
            height: size,
            width: size,
        }
    }
}

impl From<(usize, usize)> for Size2 {
    fn from((height, width): (usize, usize)) -> Size2 {
        Size2 { height, width }
    }
}

/// The geometry of a window sliding over an input plane, shared by convolution and pooling.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    pub(crate) input: Size2,
    pub(crate) kernel: Size2,
    pub(crate) output: Size2,
    pub(crate) stride: Size2,
    pub(crate) padding: Size2,
    pub(crate) dilation: Size2,
}

impl Window {
    pub(crate) fn new(
        op: &str,
        input: Size2,
        kernel: Size2,
        stride: Size2,
        padding: Size2,
        dilation: Size2,
    ) -> Window {
        let sizes = [kernel, stride, dilation];
        assert!(
            sizes.iter().all(|s| s.height > 0 && s.width > 0),
            "{op}: kernel size, stride and dilation must be positive"
        );
        let slack = |input: usize, kernel: usize, padding: usize, dilation: usize| {
            (input + 2 * padding).checked_sub(dilation * (kernel - 1) + 1)
        };
        let height = slack(input.height, kernel.height, padding.height, dilation.height);
        let width = slack(input.width, kernel.width, padding.width, dilation.width);
        let (Some(height), Some(width)) = (height, width) else {
            panic!("{op}: a kernel of {kernel:?} does not fit the padded input of {input:?}");
        };
        Window {
            input,
            kernel,
            output: Size2 {
                height: height / stride.height + 1,
                width: width / stride.width + 1,
            },
            stride,
            padding,
            dilation,
        }
    }

    /// Returns the output rows whose window reads a real input row, rather than padding, at kernel row `ky`.
    pub(crate) fn valid_rows(&self, ky: usize) -> Range<usize> {
        valid_range(
            self.input.height,
            self.output.height,
            self.stride.height,
            ky * self.dilation.height,
            self.padding.height,
        )
    }

    /// Returns the output columns whose window reads a real input column, rather than padding, at kernel column `kx`.
    pub(crate) fn valid_cols(&self, kx: usize) -> Range<usize> {
        valid_range(
            self.input.width,
            self.output.width,
            self.stride.width,
            kx * self.dilation.width,
            self.padding.width,
        )
    }

    pub(crate) fn input_area(&self) -> usize {
        self.input.height * self.input.width
    }

    pub(crate) fn output_area(&self) -> usize {
        self.output.height * self.output.width
    }
}

/// Returns the output positions `o` whose input coordinate `o * stride + offset - padding` lies inside `0..input`.
fn valid_range(
    input: usize,
    output: usize,
    stride: usize,
    offset: usize,
    padding: usize,
) -> Range<usize> {
    let start = padding.saturating_sub(offset).div_ceil(stride);
    let end = (input + padding)
        .checked_sub(offset + 1)
        .map_or(0, |last| (last / stride + 1).min(output));
    start.min(end)..end
}
/// 2-d convolution over a batch of images, like `torch.nn.functional.conv2d`.
///
/// `input` has shape `[N, C, H, W]`, `weight` has shape `[out_channels, C, kh, kw]` and `bias` has shape `[out_channels]`. Each image is unfolded into columns and multiplied with all kernels in a single matrix product.
///
/// ```
/// # use engram::{nn::functional::conv2d, Tensor};
/// let images = Tensor::randn([8, 3, 32, 32]);
/// let kernels = Tensor::randn([16, 3, 3, 3]);
/// assert_eq!(conv2d(&images, &kernels, None, 1, 1, 1).shape(), &[8, 16, 32, 32]);
/// assert_eq!(conv2d(&images, &kernels, None, 2, 0, 1).shape(), &[8, 16, 15, 15]);
/// ```
///
/// # Panics
///
/// Panics if the shapes are inconsistent or the kernel does not fit the padded input.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: impl Into<Size2>,
    padding: impl Into<Size2>,
    dilation: impl Into<Size2>,
) -> Tensor {
    let (&[batch, channels, height, width], &[out_channels, weight_channels, kh, kw]) =
        (input.shape(), weight.shape())
    else {
        panic!(
            "conv2d: expected input [N, C, H, W] and weight [out, C, kh, kw], got {:?} and {:?}",
            input.shape(),
            weight.shape()
        );
    };
    assert_eq!(
        channels, weight_channels,
        "conv2d: input has {channels} channels but the weight expects {weight_channels}"
    );
    if let Some(bias) = bias {
        assert_eq!(
            bias.shape(),
            &[out_channels],
            "conv2d: bias must have shape [{out_channels}]"
        );
    }
    let window = Window::new(
        "conv2d",
        Size2 { height, width },
        Size2 {
            height: kh,
            width: kw,
        },
        stride.into(),
        padding.into(),
        dilation.into(),
    );

    let columns = channels * kh * kw;
    let (plane, area) = (channels * window.input_area(), window.output_area());
    let (x, w, b) = (input.data(), weight.data(), bias.map(Tensor::data));
    let mut out = storage::zeros(batch * out_channels * area);
    let mut cols = storage::zeros(columns * area);
    for (n, y) in out
        .chunks_exact_mut((out_channels * area).max(1))
        .enumerate()
    {
        im2col(&x[n * plane..(n + 1) * plane], channels, &window, &mut cols);
        gemm(
            [out_channels, columns, area],
            &w,
            (columns, 1),
            &cols,
            (area, 1),
            0.0,
            y,
        );
        if let Some(b) = &b {
            for (channel, &b) in y.chunks_exact_mut(area.max(1)).zip(b.iter()) {
                channel.iter_mut().for_each(|v| *v += b);
            }
        }
    }

    let out_shape = [
        batch,
        out_channels,
        window.output.height,
        window.output.width,
    ];
    let (input_t, weight_t, bias_t) = (input.clone(), weight.clone(), bias.cloned());
    let mut inputs = vec![input, weight];
    inputs.extend(bias);
    Tensor::from_vec(out, out_shape).record(&inputs, move |grad| {
        let (g, x, w) = (grad.data(), input_t.data(), weight_t.data());
        let mut dx = input_t.requires_grad().then(|| storage::zeros(x.len()));
        let mut dw = weight_t.requires_grad().then(|| storage::zeros(w.len()));
        let mut db = vec![0.0; out_channels];
        let mut cols = storage::zeros(columns * area);
        for (n, g) in g.chunks_exact((out_channels * area).max(1)).enumerate() {
            for (db, channel) in db.iter_mut().zip(g.chunks_exact(area.max(1))) {
                *db += channel.iter().sum::<f32>();
            }
            if let Some(dw) = &mut dw {
                im2col(&x[n * plane..(n + 1) * plane], channels, &window, &mut cols);
                gemm(
                    [out_channels, area, columns],
                    g,
                    (area, 1),
                    &cols,
                    (1, area),
                    1.0,
                    dw,
                );
            }
            if let Some(dx) = &mut dx {
                gemm(
                    [columns, out_channels, area],
                    &w,
                    (1, columns),
                    g,
                    (area, 1),
                    0.0,
                    &mut cols,
                );
                col2im(
                    &cols,
                    channels,
                    &window,
                    &mut dx[n * plane..(n + 1) * plane],
                );
            }
        }

        let mut grads = vec![
            dx.map(|dx| Tensor::from_vec(dx, input_t.shape())),
            dw.map(|dw| Tensor::from_vec(dw, weight_t.shape())),
        ];
        grads.extend(
            bias_t
                .as_ref()
                .map(|b| b.requires_grad().then(|| Tensor::from_vec(db, b.shape()))),
        );
        grads
    })
}

/// Unfolds an image of shape `[channels, H, W]` into `cols` of shape `[channels * kh * kw, out_h * out_w]`, where each column holds one window.
fn im2col(image: &[f32], channels: usize, window: &Window, cols: &mut [f32]) {
    let Window {
        input,
        kernel,
        output,
        stride,
        padding,
        dilation,
    } = *window;
    let mut rows = cols.chunks_exact_mut(window.output_area().max(1));
    for channel in image
        .chunks_exact(window.input_area().max(1))
        .take(channels)
    {
        for ky in 0..kernel.height {
            let valid_rows = window.valid_rows(ky);
            for kx in 0..kernel.width {
                let valid_cols = window.valid_cols(kx);
                let first_col = (valid_cols.start * stride.width + kx * dilation.width)
                    .saturating_sub(padding.width);
                let row = rows.next().unwrap();
                for (y, out) in row.chunks_exact_mut(output.width).enumerate() {
                    if !valid_rows.contains(&y) || valid_cols.is_empty() {
                        out.fill(0.0);
                        continue;
                    }
                    let iy = y * stride.height + ky * dilation.height - padding.height;
                    let source = &channel[iy * input.width + first_col..(iy + 1) * input.width];
                    out[..valid_cols.start].fill(0.0);
                    out[valid_cols.end..].fill(0.0);
                    let dest = &mut out[valid_cols.clone()];
                    if stride.width == 1 {
                        dest.copy_from_slice(&source[..dest.len()]);
                    } else {
                        for (d, &s) in dest.iter_mut().zip(source.iter().step_by(stride.width)) {
                            *d = s;
                        }
                    }
                }
            }
        }
    }
}

/// Folds columns laid out like [`im2col`] back onto an image, adding the contributions of overlapping windows.
fn col2im(cols: &[f32], channels: usize, window: &Window, image: &mut [f32]) {
    let Window {
        input,
        kernel,
        output,
        stride,
        padding,
        dilation,
    } = *window;
    let mut rows = cols.chunks_exact(window.output_area().max(1));
    for channel in image
        .chunks_exact_mut(window.input_area().max(1))
        .take(channels)
    {
        for ky in 0..kernel.height {
            let valid_rows = window.valid_rows(ky);
            for kx in 0..kernel.width {
                let valid_cols = window.valid_cols(kx);
                let first_col = (valid_cols.start * stride.width + kx * dilation.width)
                    .saturating_sub(padding.width);
                let row = rows.next().unwrap();
                if valid_cols.is_empty() {
                    continue;
                }
                let lines = row.chunks_exact(output.width).enumerate();
                for (y, values) in lines.take(valid_rows.end).skip(valid_rows.start) {
                    let iy = y * stride.height + ky * dilation.height - padding.height;
                    let target = &mut channel[iy * input.width + first_col..(iy + 1) * input.width];
                    let values = &values[valid_cols.clone()];
                    if stride.width == 1 {
                        for (t, &v) in target.iter_mut().zip(values) {
                            *t += v;
                        }
                    } else {
                        for (t, &v) in target.iter_mut().step_by(stride.width).zip(values) {
                            *t += v;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::testing::assert_gradients;

    fn naive_conv2d(
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        [stride, padding, dilation]: [usize; 3],
    ) -> Tensor {
        let &[n, c, h, width] = x.shape() else {
            unreachable!()
        };
        let &[o, _, kh, kw] = w.shape() else {
            unreachable!()
        };
        let out_h = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
        let out_w = (width + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
        let (xs, ws, bs) = (x.to_vec(), w.to_vec(), b.to_vec());
        let mut out = Vec::new();
        for ni in 0..n {
            for oi in 0..o {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = bs[oi];
                        for ci in 0..c {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy =
                                        (oy * stride + ky * dilation) as isize - padding as isize;
                                    let ix =
                                        (ox * stride + kx * dilation) as isize - padding as isize;
                                    if (0..h as isize).contains(&iy)
                                        && (0..width as isize).contains(&ix)
                                    {
                                        let input =
                                            ((ni * c + ci) * h + iy as usize) * width + ix as usize;
                                        sum += xs[input] * ws[((oi * c + ci) * kh + ky) * kw + kx];
                                    }
                                }
                            }
                        }
                        out.push(sum);
                    }
                }
            }
        }
        Tensor::from_vec(out, [n, o, out_h, out_w])
    }

    #[test]
    fn matches_naive_convolution() {
        for options in [[1, 0, 1], [2, 1, 1], [1, 2, 2], [3, 1, 2]] {
            let [stride, padding, dilation] = options;
            let (x, w, b) = (
                Tensor::randn([2, 3, 9, 8]),
                Tensor::randn([4, 3, 3, 2]),
                Tensor::randn([4]),
            );
            let expected = naive_conv2d(&x, &w, &b, options);
            let actual = conv2d(&x, &w, Some(&b), stride, padding, dilation);
            assert!(actual.allclose(&expected, 1e-4), "options {options:?}");
        }
    }

    #[test]
    fn gradients_of_conv2d() {
        let (x, w, b) = (
            Tensor::randn([2, 2, 5, 4]),
            Tensor::randn([3, 2, 3, 2]),
            Tensor::randn([3]),
        );
        assert_gradients([&x, &w, &b], |t| conv2d(&t[0], &t[1], Some(&t[2]), 2, 1, 1));
        assert_gradients([&x, &w], |t| conv2d(&t[0], &t[1], None, (1, 2), 0, 2));
    }

    #[test]
    #[should_panic(expected = "does not fit the padded input")]
    fn rejects_kernels_larger_than_the_input() {
        conv2d(
            &Tensor::zeros([1, 1, 2, 2]),
            &Tensor::zeros([1, 1, 3, 3]),
            None,
            1,
            0,
            1,
        );
    }
}
