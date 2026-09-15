//! Elementwise math functions.

use crate::{tensor::storage::CollectBuffer, Tensor};

impl Tensor {
    /// Applies `f` to every element. `df(x, y)` returns the derivative of `f` at input `x`, where `y` is the output, so the backward pass needs a single loop.
    pub(crate) fn unary_op<F, D>(&self, f: F, df: D) -> Tensor
    where
        F: Fn(f32) -> f32,
        D: Fn(f32, f32) -> f32 + Send + Sync + 'static,
    {
        let data = self.data().iter().map(|&x| f(x)).collect_buffer();
        let out = Tensor::from_vec(data, self.shape());
        let (input, output) = (self.clone(), out.clone());
        out.record(&[self], move |grad| {
            let (g, x, y) = (grad.data(), input.data(), output.data());
            let data = g
                .iter()
                .zip(x.iter().zip(y.iter()))
                .map(|(&g, (&x, &y))| g * df(x, y))
                .collect_buffer();
            vec![Some(Tensor::from_vec(data, input.shape()))]
        })
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(|x| -x, |_, _| -1.0)
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(f32::exp, |_, y| y)
    }

    /// Natural logarithm.
    pub fn log(&self) -> Tensor {
        self.unary_op(f32::ln, |x, _| 1.0 / x)
    }

    pub fn sqrt(&self) -> Tensor {
        self.unary_op(f32::sqrt, |_, y| 0.5 / y)
    }

    pub fn square(&self) -> Tensor {
        self.unary_op(|x| x * x, |x, _| 2.0 * x)
    }

    pub fn pow(&self, exponent: f32) -> Tensor {
        self.unary_op(
            move |x| x.powf(exponent),
            move |x, _| exponent * x.powf(exponent - 1.0),
        )
    }

    pub fn abs(&self) -> Tensor {
        self.unary_op(f32::abs, |x, _| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    /// Limits every element to `[min, max]`. Gradients only flow through elements that were inside the range.
    ///
    /// # Panics
    ///
    /// Panics if `min` is greater than `max`.
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        assert!(min <= max, "clamp expects min <= max, got {min} and {max}");
        self.unary_op(
            move |x| x.clamp(min, max),
            move |x, _| f32::from((min..=max).contains(&x)),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{autograd::testing::assert_gradients, tensor, Tensor};

    #[test]
    fn computes_values() {
        let x = tensor![1, 4];
        assert_eq!(x.sqrt(), tensor![1, 2]);
        assert_eq!(x.square(), tensor![1, 16]);
        assert!(x.log().allclose(&tensor![0.0, 4f32.ln()], 1e-6));
        assert_eq!(tensor![2, 3].pow(3.0), tensor![8, 27]);
        assert_eq!(tensor![-2, 3].abs(), tensor![2, 3]);
        assert_eq!(tensor![-2, 0.5, 3].clamp(-1.0, 1.0), tensor![-1, 0.5, 1]);
    }

    #[test]
    fn gradients_of_math_ops() {
        let x = Tensor::rand([2, 3]) + 0.5;
        let ops: [fn(&Tensor) -> Tensor; 5] = [
            Tensor::neg,
            Tensor::exp,
            Tensor::log,
            Tensor::sqrt,
            Tensor::square,
        ];
        for op in ops {
            assert_gradients([&x], |t| op(&t[0]));
        }
        assert_gradients([&x], |t| t[0].pow(2.5));
    }

    #[test]
    fn gradients_skip_kinks() {
        let x = tensor![-2, -0.5, 0.5, 2];
        assert_gradients([&x], |t| t[0].abs());
        assert_gradients([&x], |t| t[0].clamp(-1.0, 1.0));
    }
}
