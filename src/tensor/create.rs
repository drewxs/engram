use crate::{
    random,
    tensor::{
        shape,
        storage::{self, CollectBuffer},
        IntoShape,
    },
    Tensor,
};

impl Tensor {
    /// Creates a tensor filled with zeros.
    ///
    /// ```
    /// # use engram::Tensor;
    /// let t = Tensor::zeros([2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn zeros(shape: impl IntoShape) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    /// Creates a tensor filled with ones.
    pub fn ones(shape: impl IntoShape) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    /// Creates a tensor filled with `value`.
    pub fn full(shape: impl IntoShape, value: f32) -> Tensor {
        let shape = shape.into_shape();
        Tensor::from_vec(storage::full(shape::numel(&shape), value), shape)
    }

    /// Creates a tensor of zeros with the same shape.
    pub fn zeros_like(&self) -> Tensor {
        Tensor::zeros(self.shape())
    }

    /// Creates a tensor of ones with the same shape.
    pub fn ones_like(&self) -> Tensor {
        Tensor::ones(self.shape())
    }

    /// Creates a tensor of the same shape filled with `value`.
    pub fn full_like(&self, value: f32) -> Tensor {
        Tensor::full(self.shape(), value)
    }

    /// Creates a 1-d tensor of values from `start` up to but excluding `end`, spaced by `step`.
    ///
    /// ```
    /// # use engram::Tensor;
    /// assert_eq!(Tensor::arange(0.0, 2.0, 0.5).to_vec(), vec![0.0, 0.5, 1.0, 1.5]);
    /// ```
    pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
        assert!(step != 0.0, "arange step must not be zero");
        let len = ((end - start) / step).ceil().max(0.0) as usize;
        let data = (0..len).map(|i| start + i as f32 * step).collect();
        Tensor::from_vec(data, [len])
    }

    /// Creates a 1-d tensor of `steps` evenly spaced values from `start` to `end`, inclusive.
    pub fn linspace(start: f32, end: f32, steps: usize) -> Tensor {
        let step = if steps > 1 {
            (end - start) / (steps - 1) as f32
        } else {
            0.0
        };
        let data = (0..steps).map(|i| start + i as f32 * step).collect();
        Tensor::from_vec(data, [steps])
    }

    /// Creates an `n` by `n` identity matrix.
    pub fn eye(n: usize) -> Tensor {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_vec(data, [n, n])
    }

    /// Creates a tensor of samples from the uniform distribution on `[0, 1)`.
    pub fn rand(shape: impl IntoShape) -> Tensor {
        Tensor::uniform(shape, 0.0, 1.0)
    }

    /// Creates a tensor of samples from the standard normal distribution.
    pub fn randn(shape: impl IntoShape) -> Tensor {
        Tensor::normal(shape, 0.0, 1.0)
    }

    /// Creates a tensor of samples from the uniform distribution on `[low, high)`.
    pub fn uniform(shape: impl IntoShape, low: f32, high: f32) -> Tensor {
        let shape = shape.into_shape();
        let n = shape::numel(&shape);
        let data = random::with_rng(|rng| (0..n).map(|_| rng.uniform(low, high)).collect_buffer());
        Tensor::from_vec(data, shape)
    }

    /// Creates a tensor of samples from a normal distribution.
    pub fn normal(shape: impl IntoShape, mean: f32, std: f32) -> Tensor {
        let shape = shape.into_shape();
        let n = shape::numel(&shape);
        let data = random::with_rng(|rng| (0..n).map(|_| rng.normal(mean, std)).collect_buffer());
        Tensor::from_vec(data, shape)
    }
}

impl From<f32> for Tensor {
    fn from(value: f32) -> Tensor {
        Tensor::scalar(value)
    }
}

impl From<&Tensor> for Tensor {
    fn from(tensor: &Tensor) -> Tensor {
        tensor.clone()
    }
}
impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Tensor {
        let len = data.len();
        Tensor::from_vec(data, [len])
    }
}

impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Tensor {
        Tensor::from_vec(data.to_vec(), [data.len()])
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(data: [f32; N]) -> Tensor {
        Tensor::from_vec(data.to_vec(), [N])
    }
}

impl<const N: usize, const M: usize> From<[[f32; M]; N]> for Tensor {
    fn from(data: [[f32; M]; N]) -> Tensor {
        Tensor::from_vec(data.as_flattened().to_vec(), [N, M])
    }
}

impl<const N: usize, const M: usize, const K: usize> From<[[[f32; K]; M]; N]> for Tensor {
    fn from(data: [[[f32; K]; M]; N]) -> Tensor {
        Tensor::from_vec(data.as_flattened().as_flattened().to_vec(), [N, M, K])
    }
}

/// Creates a [`Tensor`] from nested lists of numbers.
///
/// Values of any numeric type are converted to `f32`, and nesting depth sets the number of dimensions.
///
/// ```
/// # use engram::tensor;
/// let t = tensor![[1, 2, 3], [4, 5, 6]];
/// assert_eq!(t.shape(), &[2, 3]);
///
/// let v = tensor![0.5, -1.0];
/// assert_eq!(v.shape(), &[2]);
/// ```
#[macro_export]
macro_rules! tensor {
    (@array [$([$($inner:tt)*]),+ $(,)?]) => {
        [$($crate::tensor!(@array [$($inner)*])),+]
    };
    (@array [$($x:expr),* $(,)?]) => {
        [$($x as f32),*]
    };
    ($($t:tt)*) => {
        $crate::Tensor::from($crate::tensor!(@array [$($t)*]))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manual_seed;

    #[test]
    fn builds_nested_literals() {
        assert_eq!(tensor![1, 2].shape(), &[2]);
        assert_eq!(tensor![[1, 2], [3, 4], [5, 6]].shape(), &[3, 2]);
        let t = tensor![[[1, 2]], [[3, 4]]];
        assert_eq!(t.shape(), &[2, 1, 2]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor![-1.5, 2].to_vec(), vec![-1.5, 2.0]);
    }

    #[test]
    fn creates_ranges() {
        assert_eq!(Tensor::arange(0.0, 3.0, 1.0).to_vec(), vec![0.0, 1.0, 2.0]);
        assert_eq!(Tensor::arange(3.0, 0.0, 1.0).numel(), 0);
        assert_eq!(
            Tensor::linspace(0.0, 1.0, 5).to_vec(),
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        );
        assert_eq!(Tensor::eye(2), tensor![[1, 0], [0, 1]]);
    }

    #[test]
    fn seeded_random_tensors_are_reproducible() {
        manual_seed(11);
        let a = Tensor::randn([3, 4]);
        manual_seed(11);
        let b = Tensor::randn([3, 4]);
        assert_eq!(a, b);
        assert!(Tensor::uniform([100], -1.0, 1.0)
            .to_vec()
            .iter()
            .all(|x| (-1.0..1.0).contains(x)));
    }
}
