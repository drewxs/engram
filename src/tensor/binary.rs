//! Elementwise binary operations with broadcasting.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{tensor::shape, Tensor};

impl Tensor {
    fn add_tensor(&self, other: &Tensor) -> Tensor {
        let out = zip_broadcast(self, other, "add", |a, b| a + b);
        let (a, b) = (self.clone(), other.clone());
        out.record(&[self, other], move |g| {
            vec![
                a.requires_grad().then(|| sum_to(g, a.shape())),
                b.requires_grad().then(|| sum_to(g, b.shape())),
            ]
        })
    }

    fn sub_tensor(&self, other: &Tensor) -> Tensor {
        let out = zip_broadcast(self, other, "sub", |a, b| a - b);
        let (a, b) = (self.clone(), other.clone());
        out.record(&[self, other], move |g| {
            vec![
                a.requires_grad().then(|| sum_to(g, a.shape())),
                b.requires_grad().then(|| sum_to(&g.neg(), b.shape())),
            ]
        })
    }

    fn mul_tensor(&self, other: &Tensor) -> Tensor {
        let out = zip_broadcast(self, other, "mul", |a, b| a * b);
        let (a, b) = (self.clone(), other.clone());
        out.record(&[self, other], move |g| {
            vec![
                a.requires_grad().then(|| sum_to(&(g * &b), a.shape())),
                b.requires_grad().then(|| sum_to(&(g * &a), b.shape())),
            ]
        })
    }

    fn div_tensor(&self, other: &Tensor) -> Tensor {
        let out = zip_broadcast(self, other, "div", |a, b| a / b);
        let (a, b, y) = (self.clone(), other.clone(), out.clone());
        out.record(&[self, other], move |g| {
            let ga = g / &b;
            let gb = b.requires_grad().then(|| sum_to(&-(&ga * &y), b.shape()));
            vec![a.requires_grad().then(|| sum_to(&ga, a.shape())), gb]
        })
    }

    fn add_scalar(&self, s: f32) -> Tensor {
        self.unary_op(move |x| x + s, |_, _| 1.0)
    }

    fn sub_scalar(&self, s: f32) -> Tensor {
        self.unary_op(move |x| x - s, |_, _| 1.0)
    }

    fn rsub_scalar(&self, s: f32) -> Tensor {
        self.unary_op(move |x| s - x, |_, _| -1.0)
    }

    fn mul_scalar(&self, s: f32) -> Tensor {
        self.unary_op(move |x| x * s, move |_, _| s)
    }

    fn div_scalar(&self, s: f32) -> Tensor {
        self.unary_op(move |x| x / s, move |_, _| 1.0 / s)
    }

    fn rdiv_scalar(&self, s: f32) -> Tensor {
        self.unary_op(move |x| s / x, move |x, _| -s / (x * x))
    }

    /// Returns 1 where the elements are equal and 0 elsewhere, broadcasting both sides. The result does not track gradients.
    ///
    /// ```
    /// # use engram::tensor;
    /// let x = tensor![1, 2, 3];
    /// assert_eq!(x.equal(2.0), tensor![0, 1, 0]);
    /// ```
    pub fn equal(&self, other: impl Into<Tensor>) -> Tensor {
        zip_broadcast(self, &other.into(), "equal", |a, b| f32::from(a == b))
    }

    /// Returns 1 where `self` is greater than `other` and 0 elsewhere. The result does not track gradients.
    pub fn greater(&self, other: impl Into<Tensor>) -> Tensor {
        zip_broadcast(self, &other.into(), "greater", |a, b| f32::from(a > b))
    }

    /// Returns 1 where `self` is less than `other` and 0 elsewhere. The result does not track gradients.
    pub fn less(&self, other: impl Into<Tensor>) -> Tensor {
        zip_broadcast(self, &other.into(), "less", |a, b| f32::from(a < b))
    }
}

/// Applies `f` to every pair of elements after broadcasting both tensors to a common shape, without recording gradients.
pub(crate) fn zip_broadcast(
    a: &Tensor,
    b: &Tensor,
    op: &str,
    f: impl Fn(f32, f32) -> f32,
) -> Tensor {
    let out_shape = shape::broadcast_shape(a.shape(), b.shape()).unwrap_or_else(|| {
        panic!(
            "{op}: shapes {:?} and {:?} cannot be broadcast together",
            a.shape(),
            b.shape()
        )
    });
    let (x, y) = (a.data(), b.data());
    let data = if a.shape() == b.shape() {
        x.iter().zip(y.iter()).map(|(&x, &y)| f(x, y)).collect()
    } else if y.len() == 1 {
        x.iter().map(|&x| f(x, y[0])).collect()
    } else if x.len() == 1 {
        y.iter().map(|&y| f(x[0], y)).collect()
    } else {
        let sx = shape::broadcast_strides(a.shape(), &out_shape);
        let sy = shape::broadcast_strides(b.shape(), &out_shape);
        let inner = out_shape[out_shape.len() - 1];
        let steps = (sx[sx.len() - 1], sy[sy.len() - 1]);
        let mut data = Vec::with_capacity(shape::numel(&out_shape));
        shape::for_each_row(&out_shape, [&sx, &sy], |[i, j]| match steps {
            (1, 1) => data.extend(
                x[i..i + inner]
                    .iter()
                    .zip(&y[j..j + inner])
                    .map(|(&x, &y)| f(x, y)),
            ),
            (1, 0) => data.extend(x[i..i + inner].iter().map(|&x| f(x, y[j]))),
            (0, 1) => data.extend(y[j..j + inner].iter().map(|&y| f(x[i], y))),
            _ => data.extend((0..inner).map(|_| f(x[i], y[j]))),
        });
        data
    };
    Tensor::from_vec(data, out_shape)
}

/// Sums `grad` over the dimensions that were broadcast, so that it matches `shape`.
pub(crate) fn sum_to(grad: &Tensor, shape: &[usize]) -> Tensor {
    if grad.shape() == shape {
        return grad.clone();
    }
    let g = grad.data();
    let mut out = vec![0.0; shape::numel(shape)];
    let strides = shape::broadcast_strides(shape, grad.shape());
    let inner = grad.shape().last().copied().unwrap_or(1).max(1);
    let step = strides.last().copied().unwrap_or(0);
    let mut rows = g.chunks_exact(inner);
    shape::for_each_row(grad.shape(), [&strides], |[start]| {
        for (k, &v) in rows.next().unwrap().iter().enumerate() {
            out[start + k * step] += v;
        }
    });
    Tensor::from_vec(out, shape)
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $tensor_op:ident, $scalar_op:ident, $scalar_lhs_op:ident) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                self.$tensor_op(rhs)
            }
        }

        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                self.$tensor_op(&rhs)
            }
        }

        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                (&self).$tensor_op(rhs)
            }
        }

        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                (&self).$tensor_op(&rhs)
            }
        }

        impl $trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                self.$scalar_op(rhs)
            }
        }

        impl $trait<f32> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                (&self).$scalar_op(rhs)
            }
        }

        impl $trait<&Tensor> for f32 {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                rhs.$scalar_lhs_op(self)
            }
        }

        impl $trait<Tensor> for f32 {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                (&rhs).$scalar_lhs_op(self)
            }
        }
    };
}

impl_binary_op!(Add, add, add_tensor, add_scalar, add_scalar);
impl_binary_op!(Sub, sub, sub_tensor, sub_scalar, rsub_scalar);
impl_binary_op!(Mul, mul, mul_tensor, mul_scalar, mul_scalar);
impl_binary_op!(Div, div, div_tensor, div_scalar, rdiv_scalar);

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::sum_to;
    use crate::{autograd::testing::assert_gradients, tensor, Tensor};

    #[test]
    fn broadcasts_elementwise_ops() {
        let a = tensor![[1, 2, 3], [4, 5, 6]];
        assert_eq!(
            &a + &tensor![10, 20, 30],
            tensor![[11, 22, 33], [14, 25, 36]]
        );
        assert_eq!(
            tensor![[1], [2]] * tensor![[1, 2, 3]],
            tensor![[1, 2, 3], [2, 4, 6]]
        );
        assert_eq!(&a - Tensor::scalar(1.0), tensor![[0, 1, 2], [3, 4, 5]]);
        assert_eq!(
            (Tensor::zeros([4, 1, 3]) + Tensor::ones([2, 1])).shape(),
            &[4, 2, 3]
        );
    }

    #[test]
    fn applies_scalar_ops_on_either_side() {
        let x = tensor![1, 2, 4];
        assert_eq!(&x + 1.0, tensor![2, 3, 5]);
        assert_eq!(1.0 - &x, tensor![0, -1, -3]);
        assert_eq!(&x * 2.0, tensor![2, 4, 8]);
        assert_eq!(4.0 / &x, tensor![4, 2, 1]);
        assert_eq!(-x, tensor![-1, -2, -4]);
    }

    #[test]
    #[should_panic(expected = "add: shapes [2, 3] and [2] cannot be broadcast together")]
    fn rejects_incompatible_shapes() {
        let _ = Tensor::zeros([2, 3]) + Tensor::zeros([2]);
    }

    #[test]
    fn compares_elementwise() {
        let x = tensor![1, 2, 3];
        assert_eq!(x.equal(2.0), tensor![0, 1, 0]);
        assert_eq!(x.greater(tensor![3, 1, 2]), tensor![0, 1, 1]);
        assert_eq!(x.less(2.5), tensor![1, 1, 0]);
    }

    #[test]
    fn sums_gradients_back_to_input_shapes() {
        let g = Tensor::ones([4, 2, 3]);
        assert_eq!(sum_to(&g, &[2, 1]), tensor![[12], [12]]);
        assert_eq!(sum_to(&g, &[3]), tensor![8, 8, 8]);
        assert_eq!(sum_to(&g, &[]), Tensor::scalar(24.0));
    }

    #[test]
    fn gradients_of_broadcast_ops() {
        let a = Tensor::randn([2, 3]);
        let b = Tensor::randn([3]);
        let c = Tensor::rand([2, 1]) + 0.5;
        assert_gradients([&a, &b], |t| &t[0] + &t[1]);
        assert_gradients([&a, &b], |t| &t[0] - &t[1]);
        assert_gradients([&a, &c], |t| &t[0] * &t[1]);
        assert_gradients([&a, &c], |t| &t[0] / &t[1]);
    }

    #[test]
    fn gradients_of_scalar_ops() {
        let x = Tensor::rand([4]) + 0.5;
        assert_gradients([&x], |t| (&t[0] + 1.0) * 3.0 - 2.0);
        assert_gradients([&x], |t| 1.0 / &t[0] + (2.0 - &t[0] / 4.0));
    }
}
