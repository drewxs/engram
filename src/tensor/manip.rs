//! Reshaping, reordering, joining and slicing.

use crate::{
    tensor::{binary::sum_to, shape, Dim, IntoShape},
    Tensor,
};

impl Tensor {
    /// Returns a tensor with the same elements in a new shape. The data is shared rather than copied.
    ///
    /// # Panics
    ///
    /// Panics if the new shape holds a different number of elements.
    pub fn reshape(&self, shape: impl IntoShape) -> Tensor {
        let shape = shape.into_shape();
        assert_eq!(
            shape::numel(&shape),
            self.numel(),
            "reshape: cannot reshape {:?} into {shape:?}",
            self.shape()
        );
        let out = Tensor::from_parts(self.data(), shape);
        let input_shape = self.shape().to_vec();
        out.record(&[self], move |g| vec![Some(g.reshape(&input_shape))])
    }

    /// Merges dimensions `start` through `end`, inclusive, into one.
    ///
    /// ```
    /// # use engram::Tensor;
    /// let t = Tensor::zeros([2, 3, 4, 5]);
    /// assert_eq!(t.flatten(1, -1).shape(), &[2, 60]);
    /// ```
    pub fn flatten(&self, start: impl Dim, end: impl Dim) -> Tensor {
        if self.ndim() == 0 {
            return self.reshape([1]);
        }
        let (start, end) = (start.resolve(self.ndim()), end.resolve(self.ndim()));
        assert!(
            start <= end,
            "flatten: start dimension {start} is after end dimension {end}"
        );
        let dims = self.shape();
        let mut shape = dims[..start].to_vec();
        shape.push(shape::numel(&dims[start..=end]));
        shape.extend_from_slice(&dims[end + 1..]);
        self.reshape(shape)
    }

    /// Inserts a dimension of size one at `dim`.
    pub fn unsqueeze(&self, dim: impl Dim) -> Tensor {
        let mut shape = self.shape().to_vec();
        shape.insert(dim.resolve(self.ndim() + 1), 1);
        self.reshape(shape)
    }

    /// Removes the dimension of size one at `dim`.
    ///
    /// # Panics
    ///
    /// Panics if that dimension does not have size one.
    pub fn squeeze(&self, dim: impl Dim) -> Tensor {
        let dim = dim.resolve(self.ndim());
        assert_eq!(
            self.shape()[dim],
            1,
            "squeeze: dimension {dim} has size {}",
            self.shape()[dim]
        );
        let mut shape = self.shape().to_vec();
        shape.remove(dim);
        self.reshape(shape)
    }

    /// Reorders dimensions so that dimension `dims[i]` of the input becomes dimension `i` of the output. The data is copied.
    ///
    /// ```
    /// # use engram::Tensor;
    /// let t = Tensor::zeros([2, 3, 4]).permute(&[2, 0, 1]);
    /// assert_eq!(t.shape(), &[4, 2, 3]);
    /// ```
    pub fn permute(&self, dims: &[usize]) -> Tensor {
        let n = self.ndim();
        let mut sorted = dims.to_vec();
        sorted.sort_unstable();
        assert!(
            sorted.iter().copied().eq(0..n),
            "permute: {dims:?} is not a permutation of {n} dimensions"
        );
        if dims.iter().copied().eq(0..n) {
            return self.clone();
        }
        let strides = shape::strides(self.shape());
        let out_shape: Vec<usize> = dims.iter().map(|&d| self.shape()[d]).collect();
        let src_strides: Vec<usize> = dims.iter().map(|&d| strides[d]).collect();
        let data = gather_strided(&self.data(), &out_shape, &src_strides);
        let mut inverse = vec![0; n];
        for (i, &d) in dims.iter().enumerate() {
            inverse[d] = i;
        }
        Tensor::from_vec(data, out_shape).record(&[self], move |g| vec![Some(g.permute(&inverse))])
    }

    /// Swaps two dimensions. The data is copied.
    pub fn transpose(&self, dim0: impl Dim, dim1: impl Dim) -> Tensor {
        let mut dims: Vec<usize> = (0..self.ndim()).collect();
        dims.swap(dim0.resolve(self.ndim()), dim1.resolve(self.ndim()));
        self.permute(&dims)
    }

    /// Transposes a 2-d tensor.
    pub fn t(&self) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "t() expects a 2-d tensor, got shape {:?}",
            self.shape()
        );
        self.transpose(0, 1)
    }

    /// Repeats dimensions of size one to match `shape`. The data is copied.
    pub fn broadcast_to(&self, shape: impl IntoShape) -> Tensor {
        let shape = shape.into_shape();
        assert!(
            shape::broadcast_shape(self.shape(), &shape).is_some_and(|s| s == shape),
            "broadcast_to: cannot broadcast {:?} to {shape:?}",
            self.shape()
        );
        if self.shape() == shape {
            return self.clone();
        }
        let strides = shape::broadcast_strides(self.shape(), &shape);
        let data = gather_strided(&self.data(), &shape, &strides);
        let input_shape = self.shape().to_vec();
        Tensor::from_vec(data, shape).record(&[self], move |g| vec![Some(sum_to(g, &input_shape))])
    }

    /// Joins tensors along an existing dimension.
    ///
    /// ```
    /// # use engram::{tensor, Tensor};
    /// let t = Tensor::cat(&[tensor![[1, 2]], tensor![[3, 4]]], 0);
    /// assert_eq!(t, tensor![[1, 2], [3, 4]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `tensors` is empty or their shapes differ outside `dim`.
    pub fn cat(tensors: &[Tensor], dim: impl Dim) -> Tensor {
        let first = tensors.first().expect("cat expects at least one tensor");
        let dim = dim.resolve(first.ndim());
        for t in tensors {
            let compatible = t.ndim() == first.ndim()
                && (0..t.ndim()).all(|d| d == dim || t.shape()[d] == first.shape()[d]);
            assert!(
                compatible,
                "cat: shapes {:?} and {:?} differ outside dimension {dim}",
                first.shape(),
                t.shape()
            );
        }

        let (outer, _, inner) = shape::split_at_dim(first.shape(), dim);
        let sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[dim]).collect();
        let mut out_shape = first.shape().to_vec();
        out_shape[dim] = sizes.iter().sum();
        let parts: Vec<_> = tensors.iter().map(Tensor::data).collect();
        let mut data = Vec::with_capacity(shape::numel(&out_shape));
        for o in 0..outer {
            for (part, &size) in parts.iter().zip(&sizes) {
                data.extend_from_slice(&part[o * size * inner..(o + 1) * size * inner]);
            }
        }

        let inputs: Vec<&Tensor> = tensors.iter().collect();
        Tensor::from_vec(data, out_shape).record(&inputs, move |g| {
            let mut start = 0;
            sizes
                .iter()
                .map(|&size| {
                    let part = g.narrow(dim, start, size);
                    start += size;
                    Some(part)
                })
                .collect()
        })
    }

    /// Stacks tensors of the same shape along a new dimension.
    ///
    /// ```
    /// # use engram::{tensor, Tensor};
    /// let t = Tensor::stack(&[tensor![1, 2], tensor![3, 4]], 1);
    /// assert_eq!(t, tensor![[1, 3], [2, 4]]);
    /// ```
    pub fn stack(tensors: &[Tensor], dim: impl Dim) -> Tensor {
        let first = tensors.first().expect("stack expects at least one tensor");
        let dim = dim.resolve(first.ndim() + 1);
        let expanded: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();
        Tensor::cat(&expanded, dim)
    }

    /// Returns `len` consecutive entries along `dim`, beginning at `start`. The data is copied.
    ///
    /// # Panics
    ///
    /// Panics if the range extends past the end of the dimension.
    pub fn narrow(&self, dim: impl Dim, start: usize, len: usize) -> Tensor {
        let dim = dim.resolve(self.ndim());
        let size = self.shape()[dim];
        assert!(
            start + len <= size,
            "narrow: range {start}..{} is out of bounds for dimension {dim} of size {size}",
            start + len
        );
        let (outer, _, inner) = shape::split_at_dim(self.shape(), dim);
        let x = self.data();
        let mut data = Vec::with_capacity(outer * len * inner);
        for o in 0..outer {
            let base = (o * size + start) * inner;
            data.extend_from_slice(&x[base..base + len * inner]);
        }
        let mut out_shape = self.shape().to_vec();
        out_shape[dim] = len;

        let input_shape = self.shape().to_vec();
        Tensor::from_vec(data, out_shape).record(&[self], move |g| {
            let g = g.data();
            let mut dx = vec![0.0; shape::numel(&input_shape)];
            for o in 0..outer {
                let base = (o * size + start) * inner;
                dx[base..base + len * inner]
                    .copy_from_slice(&g[o * len * inner..(o + 1) * len * inner]);
            }
            vec![Some(Tensor::from_vec(dx, input_shape.clone()))]
        })
    }

    /// Picks one entry along `dim` and removes that dimension.
    pub fn select(&self, dim: impl Dim, index: usize) -> Tensor {
        let dim = dim.resolve(self.ndim());
        self.narrow(dim, index, 1).squeeze(dim)
    }

    /// Gathers the entries at `indices` along `dim`. The data is copied.
    ///
    /// ```
    /// # use engram::tensor;
    /// let t = tensor![[1, 2], [3, 4], [5, 6]];
    /// assert_eq!(t.index_select(0, &[2, 0]), tensor![[5, 6], [1, 2]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if an index is out of bounds.
    pub fn index_select(&self, dim: impl Dim, indices: &[usize]) -> Tensor {
        let dim = dim.resolve(self.ndim());
        let size = self.shape()[dim];
        if let Some(bad) = indices.iter().find(|&&i| i >= size) {
            panic!("index_select: index {bad} is out of bounds for dimension {dim} of size {size}");
        }
        let (outer, _, inner) = shape::split_at_dim(self.shape(), dim);
        let x = self.data();
        let mut data = Vec::with_capacity(outer * indices.len() * inner);
        for o in 0..outer {
            for &i in indices {
                let base = (o * size + i) * inner;
                data.extend_from_slice(&x[base..base + inner]);
            }
        }
        let mut out_shape = self.shape().to_vec();
        out_shape[dim] = indices.len();

        let (input_shape, indices) = (self.shape().to_vec(), indices.to_vec());
        Tensor::from_vec(data, out_shape).record(&[self], move |g| {
            let g = g.data();
            let mut dx = vec![0.0; shape::numel(&input_shape)];
            let mut offset = 0;
            for o in 0..outer {
                for &i in &indices {
                    let base = (o * size + i) * inner;
                    for (d, &v) in dx[base..base + inner]
                        .iter_mut()
                        .zip(&g[offset..offset + inner])
                    {
                        *d += v;
                    }
                    offset += inner;
                }
            }
            vec![Some(Tensor::from_vec(dx, input_shape.clone()))]
        })
    }
}

/// Copies elements of `x` in the order visited by walking `shape` with `strides`.
fn gather_strided(x: &[f32], shape: &[usize], strides: &[usize]) -> Vec<f32> {
    let inner = shape.last().copied().unwrap_or(1);
    let step = strides.last().copied().unwrap_or(0);
    let mut data = Vec::with_capacity(shape::numel(shape));
    shape::for_each_row(shape, [strides], |[start]| {
        if step == 1 {
            data.extend_from_slice(&x[start..start + inner]);
        } else {
            data.extend((0..inner).map(|k| x[start + k * step]));
        }
    });
    data
}

#[cfg(test)]
mod tests {
    use crate::{autograd::testing::assert_gradients, tensor, Tensor};

    #[test]
    fn reshapes_without_copying() {
        let t = Tensor::arange(0.0, 6.0, 1.0);
        let r = t.reshape([2, 3]);
        assert_eq!(r, tensor![[0, 1, 2], [3, 4, 5]]);
        assert_eq!(r.unsqueeze(0).shape(), &[1, 2, 3]);
        assert_eq!(r.unsqueeze(-1).shape(), &[2, 3, 1]);
        assert_eq!(r.unsqueeze(0).squeeze(0).shape(), &[2, 3]);
        assert_eq!(Tensor::scalar(1.0).flatten(0, -1).shape(), &[1]);
    }

    #[test]
    fn permutes_dimensions() {
        let t = Tensor::arange(0.0, 6.0, 1.0).reshape([2, 3]);
        assert_eq!(t.t(), tensor![[0, 3], [1, 4], [2, 5]]);
        let u = Tensor::arange(0.0, 24.0, 1.0).reshape([2, 3, 4]);
        let p = u.permute(&[2, 0, 1]);
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.to_vec()[..6], [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]);
        assert_eq!(p.permute(&[1, 2, 0]), u);
    }

    #[test]
    fn broadcasts_to_larger_shapes() {
        let t = tensor![[1], [2]].broadcast_to([3, 2, 2]);
        assert_eq!(t.shape(), &[3, 2, 2]);
        assert_eq!(t.to_vec()[..4], [1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn joins_and_slices() {
        let a = tensor![[1, 2], [3, 4]];
        let b = tensor![[5, 6]];
        assert_eq!(
            Tensor::cat(&[a.clone(), b], 0),
            tensor![[1, 2], [3, 4], [5, 6]]
        );
        assert_eq!(
            Tensor::stack(&[a.clone(), a.clone()], 0).shape(),
            &[2, 2, 2]
        );
        assert_eq!(a.narrow(1, 1, 1), tensor![[2], [4]]);
        assert_eq!(a.select(0, 1), tensor![3, 4]);
        assert_eq!(a.index_select(1, &[1, 1, 0]), tensor![[2, 2, 1], [4, 4, 3]]);
    }

    #[test]
    #[should_panic(expected = "reshape: cannot reshape [2, 3] into [4]")]
    fn rejects_bad_reshape() {
        Tensor::zeros([2, 3]).reshape([4]);
    }

    #[test]
    fn gradients_of_shape_ops() {
        let x = Tensor::randn([2, 3, 4]);
        let y = Tensor::randn([2, 1, 4]);
        assert_gradients([&x], |t| t[0].reshape([6, 4]));
        assert_gradients([&x], |t| t[0].permute(&[1, 2, 0]));
        assert_gradients([&x], |t| t[0].transpose(0, -1));
        assert_gradients([&y], |t| t[0].broadcast_to([5, 2, 3, 4]));
        assert_gradients([&x, &y], |t| Tensor::cat(t, 1));
        assert_gradients([&y, &y], |t| Tensor::stack(t, 0));
        assert_gradients([&x], |t| t[0].narrow(1, 1, 2));
        assert_gradients([&x], |t| t[0].index_select(1, &[2, 0, 2]));
    }
}
