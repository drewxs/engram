//! Reductions over every element or along a dimension.

use crate::{
    tensor::{shape, Dim},
    Tensor,
};

impl Tensor {
    /// Sums every element into a scalar.
    pub fn sum(&self) -> Tensor {
        let total: f64 = self.data().iter().map(|&x| x as f64).sum();
        let out = Tensor::scalar(total as f32);
        let shape = self.shape().to_vec();
        out.record(&[self], move |g| vec![Some(Tensor::full(&shape, g.item()))])
    }

    /// Averages every element into a scalar.
    pub fn mean(&self) -> Tensor {
        let n = self.numel() as f32;
        let total: f64 = self.data().iter().map(|&x| x as f64).sum();
        let out = Tensor::scalar((total / n as f64) as f32);
        let shape = self.shape().to_vec();
        out.record(&[self], move |g| {
            vec![Some(Tensor::full(&shape, g.item() / n))]
        })
    }

    /// Sums along `dim`. With `keepdim` the reduced dimension stays as size one.
    ///
    /// ```
    /// # use engram::tensor;
    /// let t = tensor![[1, 2], [3, 4]];
    /// assert_eq!(t.sum_dim(0, false), tensor![4, 6]);
    /// assert_eq!(t.sum_dim(-1, true), tensor![[3], [7]]);
    /// ```
    pub fn sum_dim(&self, dim: impl Dim, keepdim: bool) -> Tensor {
        let dim = dim.resolve(self.ndim());
        let data = sum_along(&self.data(), self.shape(), dim);
        let out = Tensor::from_vec(data, reduced_shape(self.shape(), dim, keepdim));
        let input_shape = self.shape().to_vec();
        let kept_shape = reduced_shape(&input_shape, dim, true);
        out.record(&[self], move |g| {
            vec![Some(g.reshape(&kept_shape).broadcast_to(&input_shape))]
        })
    }

    /// Averages along `dim`. With `keepdim` the reduced dimension stays as size one.
    pub fn mean_dim(&self, dim: impl Dim, keepdim: bool) -> Tensor {
        let dim = dim.resolve(self.ndim());
        self.sum_dim(dim, keepdim) / self.shape()[dim] as f32
    }

    /// Returns the largest values along `dim` and the indices where they occur.
    ///
    /// ```
    /// # use engram::tensor;
    /// let (values, indices) = tensor![[1, 5], [7, 2]].max_dim(1, false);
    /// assert_eq!(values, tensor![5, 7]);
    /// assert_eq!(indices, tensor![1, 0]);
    /// ```
    pub fn max_dim(&self, dim: impl Dim, keepdim: bool) -> (Tensor, Tensor) {
        self.extreme_dim(dim.resolve(self.ndim()), keepdim, |a, b| a > b)
    }

    /// Returns the smallest values along `dim` and the indices where they occur.
    pub fn min_dim(&self, dim: impl Dim, keepdim: bool) -> (Tensor, Tensor) {
        self.extreme_dim(dim.resolve(self.ndim()), keepdim, |a, b| a < b)
    }

    /// Returns the indices of the largest values along `dim`.
    pub fn argmax(&self, dim: impl Dim, keepdim: bool) -> Tensor {
        self.max_dim(dim, keepdim).1
    }

    /// Returns the indices of the smallest values along `dim`.
    pub fn argmin(&self, dim: impl Dim, keepdim: bool) -> Tensor {
        self.min_dim(dim, keepdim).1
    }

    /// Returns the largest element as a scalar.
    pub fn max(&self) -> Tensor {
        self.reshape([self.numel()]).max_dim(0, false).0
    }

    /// Returns the smallest element as a scalar.
    pub fn min(&self) -> Tensor {
        self.reshape([self.numel()]).min_dim(0, false).0
    }

    fn extreme_dim(
        &self,
        dim: usize,
        keepdim: bool,
        better: fn(f32, f32) -> bool,
    ) -> (Tensor, Tensor) {
        let size = self.shape()[dim];
        assert!(size > 0, "cannot reduce over an empty dimension");
        let x = self.data();
        let mut values = Vec::new();
        let mut indices = Vec::new();
        shape::for_each_lane(self.shape(), dim, |start, stride| {
            let (index, value) = (0..size)
                .map(|k| (k, x[start + k * stride]))
                .reduce(|best, next| if better(next.1, best.1) { next } else { best })
                .unwrap();
            values.push(value);
            indices.push(index);
        });

        let out_shape = reduced_shape(self.shape(), dim, keepdim);
        let index_tensor = Tensor::from_vec(
            indices.iter().map(|&i| i as f32).collect(),
            out_shape.clone(),
        );
        let input_shape = self.shape().to_vec();
        let out = Tensor::from_vec(values, out_shape).record(&[self], move |g| {
            let g = g.data();
            let mut dx = vec![0.0; shape::numel(&input_shape)];
            let mut j = 0;
            shape::for_each_lane(&input_shape, dim, |start, stride| {
                dx[start + indices[j] * stride] += g[j];
                j += 1;
            });
            vec![Some(Tensor::from_vec(dx, input_shape.clone()))]
        });
        (out, index_tensor)
    }
}

fn reduced_shape(shape: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    let mut shape = shape.to_vec();
    if keepdim {
        shape[dim] = 1;
    } else {
        shape.remove(dim);
    }
    shape
}

fn sum_along(x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let (outer, size, inner) = shape::split_at_dim(shape, dim);
    let mut out = vec![0.0; outer * inner];
    for (o, dst) in out.chunks_exact_mut(inner.max(1)).enumerate() {
        for s in 0..size {
            let start = (o * size + s) * inner;
            for (d, &v) in dst.iter_mut().zip(&x[start..start + inner]) {
                *d += v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::{autograd::testing::assert_gradients, tensor, Tensor};

    #[test]
    fn reduces_all_elements() {
        let t = tensor![[1, 2], [3, 6]];
        assert_eq!(t.sum(), Tensor::scalar(12.0));
        assert_eq!(t.mean(), Tensor::scalar(3.0));
        assert_eq!(t.max(), Tensor::scalar(6.0));
        assert_eq!(t.min(), Tensor::scalar(1.0));
    }

    #[test]
    fn reduces_along_dims() {
        let t = Tensor::arange(0.0, 24.0, 1.0).reshape([2, 3, 4]);
        assert_eq!(t.sum_dim(1, false).shape(), &[2, 4]);
        assert_eq!(t.sum_dim(1, false).to_vec()[..2], [12.0, 15.0]);
        assert_eq!(t.mean_dim(-1, true).shape(), &[2, 3, 1]);
        assert_eq!(t.mean_dim(-1, true).to_vec()[..2], [1.5, 5.5]);
        assert_eq!(t.argmax(0, false), Tensor::ones([3, 4]));
        assert_eq!(tensor![[3, 1], [0, 2]].argmin(1, true), tensor![[1], [0]]);
    }

    #[test]
    fn gradients_of_reductions() {
        let x = Tensor::randn([2, 3, 4]);
        // Distinct, well separated values keep finite differences from switching the selected element.
        let mut values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        crate::random::shuffle(&mut values);
        let separated = Tensor::from_vec(values, [2, 3, 4]);
        assert_gradients([&x], |t| t[0].sum());
        assert_gradients([&x], |t| t[0].mean());
        for dim in [0, 1, 2] {
            assert_gradients([&x], |t| t[0].sum_dim(dim, false));
            assert_gradients([&x], |t| t[0].mean_dim(dim, true));
            assert_gradients([&separated], |t| t[0].max_dim(dim, false).0);
            assert_gradients([&separated], |t| t[0].min_dim(dim, true).0);
        }
    }

    #[test]
    fn autograd_through_sum() {
        let w = tensor![1, 2, 3].set_requires_grad(true);
        (&w * &w).sum().backward();
        assert_eq!(w.grad().unwrap(), tensor![2, 4, 6]);
    }
}
