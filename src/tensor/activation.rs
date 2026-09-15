//! Activation functions.

use std::f64::consts::{PI, SQRT_2};

use crate::{
    tensor::{shape, Dim},
    Tensor,
};

impl Tensor {
    pub fn relu(&self) -> Tensor {
        self.unary_op(|x| if x > 0.0 { x } else { 0.0 }, |x, _| f32::from(x > 0.0))
    }

    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor {
        self.unary_op(
            move |x| if x > 0.0 { x } else { negative_slope * x },
            move |x, _| if x > 0.0 { 1.0 } else { negative_slope },
        )
    }

    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(sigmoid, |_, y| y * (1.0 - y))
    }

    pub fn tanh(&self) -> Tensor {
        self.unary_op(f32::tanh, |_, y| 1.0 - y * y)
    }

    /// Gaussian error linear unit, using the exact error function form.
    pub fn gelu(&self) -> Tensor {
        self.unary_op(
            |x| x * normal_cdf(x),
            |x, _| normal_cdf(x) + x * normal_pdf(x),
        )
    }

    /// Normalizes values along `dim` into probabilities that sum to one.
    ///
    /// ```
    /// # use engram::tensor;
    /// let p = tensor![[1, 1], [0, 0]].softmax(-1);
    /// assert_eq!(p, tensor![[0.5, 0.5], [0.5, 0.5]]);
    /// ```
    pub fn softmax(&self, dim: impl Dim) -> Tensor {
        self.softmax_impl(dim.resolve(self.ndim()), false)
    }

    /// Computes the logarithm of [`Tensor::softmax`] without the precision loss of taking them separately.
    pub fn log_softmax(&self, dim: impl Dim) -> Tensor {
        self.softmax_impl(dim.resolve(self.ndim()), true)
    }

    fn softmax_impl(&self, dim: usize, log: bool) -> Tensor {
        let size = self.shape()[dim];
        let x = self.data();
        let mut data = vec![0.0; x.len()];
        shape::for_each_lane(self.shape(), dim, |start, stride| {
            let lane = (0..size).map(move |k| start + k * stride);
            let max = lane.clone().map(|i| x[i]).fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = max + lane.clone().map(|i| (x[i] - max).exp()).sum::<f32>().ln();
            for i in lane {
                let log_p = x[i] - log_sum_exp;
                data[i] = if log { log_p } else { log_p.exp() };
            }
        });

        let out = Tensor::from_vec(data, self.shape());
        let (shape, output) = (self.shape().to_vec(), out.clone());
        out.record(&[self], move |grad| {
            let (g, y) = (grad.data(), output.data());
            let mut dx = vec![0.0; g.len()];
            shape::for_each_lane(&shape, dim, |start, stride| {
                let lane = (0..size).map(move |k| start + k * stride);
                if log {
                    let sum: f32 = lane.clone().map(|i| g[i]).sum();
                    for i in lane {
                        dx[i] = g[i] - y[i].exp() * sum;
                    }
                } else {
                    let dot: f32 = lane.clone().map(|i| g[i] * y[i]).sum();
                    for i in lane {
                        dx[i] = y[i] * (g[i] - dot);
                    }
                }
            });
            vec![Some(Tensor::from_vec(dx, shape.clone()))]
        })
    }
}

/// Logistic sigmoid that avoids overflow for large negative inputs.
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn normal_cdf(x: f32) -> f32 {
    (0.5 * (1.0 + erf(x as f64 / SQRT_2))) as f32
}

fn normal_pdf(x: f32) -> f32 {
    ((-0.5 * (x as f64).powi(2)).exp() / (2.0 * PI).sqrt()) as f32
}

/// Error function from Abramowitz and Stegun formula 7.1.26, accurate to 1.5e-7.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (1.0 - poly * (-x * x).exp()).copysign(x)
}

#[cfg(test)]
mod tests {
    use crate::{autograd::testing::assert_gradients, tensor, Tensor};

    #[test]
    fn computes_values() {
        assert_eq!(tensor![-1, 0, 2].relu(), tensor![0, 0, 2]);
        assert_eq!(tensor![-2, 3].leaky_relu(0.1), tensor![-0.2, 3]);
        assert_eq!(tensor![0].sigmoid(), tensor![0.5]);
        assert_eq!(tensor![-1000, 1000].sigmoid(), tensor![0, 1]);
        assert!(tensor![1].gelu().allclose(&tensor![0.841_344_7], 1e-5));
    }

    #[test]
    fn softmax_is_normalized_and_stable() {
        let x = tensor![[1, 2, 3], [1000, 1000, 1000]];
        let p = x.softmax(-1);
        let third = 1.0 / 3.0;
        let expected = tensor![
            [0.090_030_57, 0.244_728_48, 0.665_240_94],
            [third, third, third]
        ];
        assert!(p.allclose(&expected, 1e-5), "{p}");
        assert!(x.log_softmax(1).allclose(&p.log(), 1e-5));
    }

    #[test]
    fn gradients_of_smooth_activations() {
        let x = Tensor::randn([2, 3, 4]);
        let ops: [fn(&Tensor) -> Tensor; 3] = [Tensor::sigmoid, Tensor::tanh, Tensor::gelu];
        for op in ops {
            assert_gradients([&x], |t| op(&t[0]));
        }
        for dim in [0, 1, -1] {
            assert_gradients([&x], |t| t[0].softmax(dim));
            assert_gradients([&x], |t| t[0].log_softmax(dim));
        }
    }

    #[test]
    fn gradients_of_piecewise_activations() {
        let x = tensor![-2, -0.5, 0.5, 2];
        assert_gradients([&x], |t| t[0].relu());
        assert_gradients([&x], |t| t[0].leaky_relu(0.2));
    }
}
