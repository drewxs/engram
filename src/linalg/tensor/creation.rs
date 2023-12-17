use crate::{Initializer, Tensor, Tensor2D};

impl Tensor {
    /// Creates a new `Tensor` of 0s with the specified number of rows and columns.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let tensor = Tensor::zeros(2, 3);
    /// let tensor_b = Tensor::zeros(3, 2);
    /// assert_eq!(tensor.data, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
    /// assert_eq!(tensor_b.data, vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]]);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Tensor {
        Tensor {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    /// Creates a new `Tensor` of 0s with the same shape as the provided `Tensor`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let other = Tensor::zeros(2, 3);
    /// let tensor = Tensor::zeros_like(&other);
    /// assert_eq!(tensor.data, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
    /// ```
    pub fn zeros_like(other: &Tensor) -> Tensor {
        Tensor::zeros(other.rows, other.cols)
    }

    /// Creates a new `Tensor` of 1s with the specified number of rows and columns.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let tensor = Tensor::ones(2, 3);
    /// let tensor_b = Tensor::ones(3, 2);
    /// assert_eq!(tensor.data, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
    /// assert_eq!(tensor_b.data, vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]]);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Tensor {
        Tensor {
            rows,
            cols,
            data: vec![vec![1.0; cols]; rows],
        }
    }

    /// Creates a new `Tensor` of 1s with the same shape as the provided `Tensor`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let other = Tensor::ones(2, 3);
    /// let tensor = Tensor::ones_like(&other);
    /// assert_eq!(tensor.data, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
    /// ```
    pub fn ones_like(other: &Tensor) -> Tensor {
        Tensor::ones(other.rows, other.cols)
    }

    /// Creates a new `Tensor` from a two-dimensional vector of floating point values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let tensor = Tensor::from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    /// assert_eq!(tensor.data, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    /// ```
    pub fn from(data: Tensor2D) -> Tensor {
        Tensor {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    /// Creates a new `Tensor` with the specified number of rows and columns, initialized using the provided initializer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let tensor = Tensor::initialize(2, 3, &Initializer::Xavier);
    /// assert_eq!(tensor.data.len(), 2);
    /// assert_eq!(tensor.data[0].len(), 3);
    /// assert!(tensor.data.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
    /// ```
    pub fn initialize(rows: usize, cols: usize, initializer: &Initializer) -> Tensor {
        let mut res = Tensor::zeros(rows, cols);
        res.data = initializer.initialize(rows, cols);
        res
    }
}
