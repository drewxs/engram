//! This module provides a Tensor struct that represents a two-dimensional matrix of
//! floating point values. It also provides type aliases for one- and two-dimensional
//! vectors of floating point values, as well as methods for initializing, manipulating,
//! and performing mathematical operations on tensors.

use std::{collections::HashMap, fmt};

use crate::initializer::Initializer;

/// A one-dimensional matrix of floating point values.
pub type Tensor1D = Vec<f64>;
/// A two-dimensional matrix of floating point values.
pub type Tensor2D = Vec<Tensor1D>;

/// A matrix of floating point values, represented as a two-dimensional vector.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The number of rows in the matrix.
    pub rows: usize,
    /// The number of columns in the matrix.
    pub cols: usize,
    /// The data in the matrix, represented as a two-dimensional vector.
    pub data: Tensor2D,
}

impl Tensor {
    /// Creates a new `Tensor` of 0s with the specified number of rows and columns.
    ///
    /// # Examples
    /// ```
    /// # use engram::Tensor;
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
    /// # use engram::Tensor;
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
    /// # use engram::Tensor;
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

    /// Creates a new `Tensor` from a two-dimensional vector of floating point values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let tensor = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
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
    /// # use engram::{Initializer, Tensor};
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

    /// Returns the shape of the tensor as a tuple of (rows, columns).
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::Tensor;
    /// let tensor = Tensor::zeros(2, 3);
    /// let tensor_b = Tensor::zeros(3, 2);
    /// assert_eq!(tensor.shape(), (2, 3));
    /// assert_eq!(tensor_b.shape(), (3, 2));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::Tensor;
    /// let tensor = Tensor::zeros(2, 3);
    /// let tensor_b = Tensor::zeros(5, 4);
    /// assert_eq!(tensor.size(), 6);
    /// assert_eq!(tensor_b.size(), 20);
    /// ```
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    /// Returns the first element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let tensor = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// assert_eq!(tensor.first(), 1.0);
    /// ```
    pub fn first(&self) -> f64 {
        self.data[0][0]
    }

    /// Returns an iterator over the rows of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut tensor = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let mut iter = tensor.iter_rows();
    /// assert_eq!(iter.next(), Some(&vec![1.0, 2.0, 3.0]));
    /// assert_eq!(iter.next(), Some(&vec![4.0, 5.0, 6.0]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_rows(&self) -> impl Iterator<Item = &Tensor1D> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the rows of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut tensor = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let mut iter = tensor.iter_rows_mut();
    /// assert_eq!(iter.next(), Some(&mut vec![1.0, 2.0, 3.0]));
    /// assert_eq!(iter.next(), Some(&mut vec![4.0, 5.0, 6.0]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut Tensor1D> {
        self.data.iter_mut()
    }

    /// Performs matrix multiplication between two tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let b = tensor![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    /// let c = a.matmul(&b);
    /// assert_eq!(c.data, vec![vec![58.0, 64.0], vec![139.0, 154.0]]);
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        self.validate_mul_shape(other, "matmul");

        let mut res = Tensor::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                res.data[i][j] = sum;
            }
        }
        res
    }

    /// Performs sparse matrix multiplication between two tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]];
    /// let b = tensor![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    /// let c = a.sparse_matmul(&b);
    /// assert_eq!(c.data, vec![vec![7.0, 8.0], vec![45.0, 50.0], vec![22.0, 24.0]]);
    /// ```
    pub fn sparse_matmul(&self, other: &Tensor) -> Tensor {
        self.validate_mul_shape(other, "matmul");

        let sparse_self = self.non_zero_values();
        let sparse_other = other.non_zero_values();

        let mut res = Tensor::zeros(self.rows, other.cols);
        for &(i, j) in sparse_self.keys() {
            for k in 0..other.cols {
                if let (Some(self_value), Some(other_value)) =
                    (sparse_self.get(&(i, j)), sparse_other.get(&(j, k)))
                {
                    res.data[i][k] = self_value * other_value;
                }
            }
        }
        res
    }

    /// Performs matrix multiplication between two tensors in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.matmul_assign(&b);
    /// assert_eq!(a.data, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    /// ```
    pub fn matmul_assign(&mut self, other: &Tensor) {
        self.validate_mul_shape(other, "matmul_assign");

        let mut res = Tensor::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                res.data[i][j] = sum;
            }
        }
        *self = res;
    }

    /// Multiplies two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a.mul(&b);
    /// assert_eq!(c.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
    /// ```
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.validate_shape(other, "mul");

        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        res
    }

    /// Multiplies two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.mul_assign(&b);
    /// assert_eq!(a.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
    /// ```
    pub fn mul_assign(&mut self, other: &Tensor) {
        self.validate_mul_shape(other, "mul_assign");

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] *= other.data[i][j];
            }
        }
    }

    /// Multiplies a tensor by a scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.mul_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    /// ```
    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * scalar;
            }
        }
        res
    }

    /// Multiplies a tensor by a scalar in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.mul_scalar_assign(5.0);
    /// assert_eq!(a.data, vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    /// ```
    pub fn mul_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] *= scalar;
            }
        }
    }

    /// Divides two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 8.0], [30.0, 8.0]];
    /// let c = a.div(&b);
    /// assert_eq!(c.data, vec![vec![0.2, 0.25], vec![0.1, 0.5]]);
    /// ```
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.validate_shape(other, "div");

        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] / other.data[i][j];
            }
        }
        res
    }

    /// Divides two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 8.0], [30.0, 8.0]];
    /// a.div_assign(&b);
    /// assert_eq!(a.data, vec![vec![0.2, 0.25], vec![0.1, 0.5]]);
    /// ```
    pub fn div_assign(&mut self, other: &Tensor) {
        self.validate_shape(other, "div_assign");

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] /= other.data[i][j];
            }
        }
    }

    /// Divides a tensor by a scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.div_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![0.2, 0.4], vec![0.6, 0.8]]);
    /// ```
    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] / scalar;
            }
        }
        res
    }

    /// Divides a tensor by a scalar in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.div_scalar_assign(5.0);
    /// assert_eq!(a.data, vec![vec![0.2, 0.4], vec![0.6, 0.8]]);
    /// ```
    pub fn div_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] /= scalar;
            }
        }
    }

    /// Adds two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a.add(&b);
    /// assert_eq!(c.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    /// ```
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.validate_shape(other, "add");

        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        res
    }

    /// Adds two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.add_assign(&b);
    /// assert_eq!(a.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    /// ```
    pub fn add_assign(&mut self, other: &Tensor) {
        self.validate_shape(other, "add_assign");

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] += other.data[i][j];
            }
        }
    }

    /// Adds a scalar to a tensor element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.add_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
    /// ```
    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + scalar;
            }
        }
        res
    }

    /// Adds a scalar to a tensor element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.add_scalar_assign(5.0);
    /// assert_eq!(a.data, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
    /// ```
    pub fn add_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] += scalar;
            }
        }
    }

    /// Subtracts two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a.sub(&b);
    /// assert_eq!(c.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.validate_shape(other, "sub");

        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        res
    }

    /// Subtracts two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.sub_assign(&b);
    /// assert_eq!(a.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
    /// ```
    pub fn sub_assign(&mut self, other: &Tensor) {
        self.validate_shape(other, "sub_assign");

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] -= other.data[i][j];
            }
        }
    }

    /// Subtracts a scalar from a tensor element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.sub_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![-4.0, -3.0], vec![-2.0, -1.0]]);
    /// ```
    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - scalar;
            }
        }
        res
    }

    /// Subtracts a scalar from a tensor element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.sub_scalar_assign(5.0);
    /// assert_eq!(a.data, vec![vec![-4.0, -3.0], vec![-2.0, -1.0]]);
    /// ```
    pub fn sub_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] -= scalar;
            }
        }
    }

    /// Applies a function to each element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.mapv(&|x| x * 2.0);
    /// assert_eq!(b.data, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    /// ```
    pub fn mapv(&self, function: &dyn Fn(f64) -> f64) -> Tensor {
        let data = (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| function(x)).collect())
            .collect::<Tensor2D>();

        Tensor {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Applies a function to each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.mapv_assign(&|x| x * 2.0);
    /// assert_eq!(a.data, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    /// ```
    pub fn mapv_assign(&mut self, function: &dyn Fn(f64) -> f64) {
        self.data = (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| function(x)).collect())
            .collect::<Tensor2D>();
    }

    /// Returns the square of each element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.square();
    /// assert_eq!(b.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn square(&self) -> Tensor {
        self.mapv(&|x| x * x)
    }

    /// Squares each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.square_assign();
    /// assert_eq!(a.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn square_assign(&mut self) {
        self.mapv_assign(&|x| x * x);
    }

    /// Returns the square root of each element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 4.0], [9.0, 16.0]];
    /// let b = a.sqrt();
    /// assert_eq!(b.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// ```
    pub fn sqrt(&self) -> Tensor {
        self.mapv(&|x| x.sqrt())
    }

    /// Takes the square root of each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 4.0], [9.0, 16.0]];
    /// a.sqrt_assign();
    /// assert_eq!(a.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// ```
    pub fn sqrt_assign(&mut self) {
        self.mapv_assign(&|x| x.sqrt());
    }

    /// Returns each element in the tensor raised to the given exponent.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.pow(2.0);
    /// assert_eq!(b.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn pow(&self, exponent: f64) -> Tensor {
        self.mapv(&|x| x.powf(exponent))
    }

    /// Raises each element in the tensor to the given exponent in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.pow_assign(2.0);
    /// assert_eq!(a.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn pow_assign(&mut self, exponent: f64) {
        self.mapv_assign(&|x| x.powf(exponent));
    }

    /// Returns each element in the tensor applied with the natural logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.ln();
    /// assert_eq!(b.data, vec![vec![0.0, 0.6931471805599453], vec![1.0986122886681098, 1.3862943611198906]]);
    /// ```
    pub fn ln(&self) -> Tensor {
        self.mapv(&|x| x.ln())
    }

    /// Applies the natural logarithm to each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.ln_assign();
    /// assert_eq!(a.data, vec![vec![0.0, 0.6931471805599453], vec![1.0986122886681098, 1.3862943611198906]]);
    /// ```
    pub fn ln_assign(&mut self) {
        self.mapv_assign(&|x| x.ln());
    }

    /// Returns each element in the tensor applied with the base 2 logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [4.0, 8.0]];
    /// let b = a.log2();
    /// assert_eq!(b.data, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    /// ```
    pub fn log2(&self) -> Tensor {
        self.mapv(&|x| x.log2())
    }

    /// Applies the base 2 logarithm to each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [4.0, 8.0]];
    /// a.log2_assign();
    /// assert_eq!(a.data, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    /// ```
    pub fn log2_assign(&mut self) {
        self.mapv_assign(&|x| x.log2());
    }

    /// Returns a flattened version of the tensor data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.flatten();
    /// assert_eq!(b, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn flatten(&self) -> Tensor1D {
        let mut flat_data = Vec::new();
        for row in &self.data {
            flat_data.extend(row);
        }
        Tensor1D::from(flat_data)
    }

    /// Returns the transpose of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]];
    /// let b = a.transpose();
    /// assert_eq!(b.data, vec![vec![1.0, 3.0], vec![2.0, 4.0], vec![3.0, 6.0]]);
    /// ```
    pub fn transpose(&self) -> Tensor {
        let mut res = Tensor::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }

    /// Transposes the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.transpose_assign();
    /// assert_eq!(a.data, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    /// ```
    pub fn transpose_assign(&mut self) {
        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        self.data = res.data;
    }

    /// Broadcasts the tensor to a new shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]];
    /// let c = a.broadcast_to(&b);
    /// assert_eq!(b.data, vec![vec![1.0, 2.0, 8.0], vec![3.0, 4.0, 9.0]]);
    /// ```
    pub fn broadcast_to(&self, other: &Tensor) -> Tensor {
        let mut result = Tensor::zeros(other.rows, other.cols);
        for i in 0..other.rows {
            for j in 0..other.cols {
                result.data[i][j] = self.data[i % self.rows][j % self.cols];
            }
        }
        result
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
    /// let b = a.reshape(2, 3);
    /// let c = tensor![[1.0, 2.0], [4.0, 5.0], [7.0, 8.0], [3.0, 6.0]];
    /// let d = c.reshape(2, 4);
    /// assert_eq!(b.data, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    /// assert_eq!(d.data, vec![vec![1.0, 2.0, 4.0, 5.0], vec![7.0, 8.0, 3.0, 6.0]]);
    /// ```
    pub fn reshape(&self, rows: usize, cols: usize) -> Tensor {
        if self.rows * self.cols != rows * cols {
            panic!("New shape must have the same total number of elements as the original tensor.");
        }

        let mut res = Tensor::zeros(rows, cols);
        let mut idx = 0;
        let flat = self.flatten();
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = flat[idx];
                idx += 1;
            }
        }
        res
    }

    /// Returns a slice of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let b = a.slice(1, 3);
    /// assert_eq!(b.data, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
    /// ```
    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        Tensor {
            rows: end - start,
            cols: self.cols,
            data: self.data[start..end].to_vec(),
        }
    }

    /// Computes the dot product between two tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0], [2.0], [3.0]];
    /// let b = tensor![[4.0], [5.0], [6.0]];
    /// let c = a.dot(&b);
    /// assert_eq!(c, 32.0);
    /// ```
    pub fn dot(&self, other: &Tensor) -> f64 {
        self.validate_shape(other, "dot");

        let flat_self = self.flatten();
        let flat_other = other.flatten();
        let mut res = 0.0;
        for i in 0..flat_self.len() {
            res += flat_self[i] * flat_other[i];
        }
        res
    }

    /// Returns the sum of all elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.sum();
    /// assert_eq!(b, 10.0);
    /// ```
    pub fn sum(&self) -> f64 {
        self.data.iter().flatten().fold(0.0, |acc, x| acc + x)
    }

    /// Returns the sum of all elements in the tensor along the given axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let sum_cols = a.sum_axis(0);
    /// let sum_rows = a.sum_axis(1);
    /// assert_eq!(sum_cols.data, vec![vec![9.0, 12.0]]);
    /// assert_eq!(sum_rows.data, vec![vec![3.0], vec![7.0], vec![11.0]]);
    /// ```
    pub fn sum_axis(&self, axis: u8) -> Tensor {
        if axis == 0 {
            let mut res = Tensor::zeros(1, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    res.data[0][j] += self.data[i][j];
                }
            }
            res
        } else if axis == 1 {
            let mut res = Tensor::zeros(self.rows, 1);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    res.data[i][0] += self.data[i][j];
                }
            }
            res
        } else {
            panic!("Invalid axis value: {}", axis);
        }
    }

    /// Returns the mean of all elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.mean();
    /// assert_eq!(b, 2.5);
    /// ```
    pub fn mean(&self) -> f64 {
        self.sum() / (self.rows * self.cols) as f64
    }

    /// Returns the p-norm of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.norm(2.0);
    /// assert_eq!(b, 5.477225575051661);
    /// ```
    pub fn norm(&self, p: f64) -> f64 {
        self.mapv(&|x| x.powf(p)).sum().sqrt()
    }

    /// Returns a vector of the indices of all non-zero elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use engram::tensor;
    /// let a = tensor![[0.0, 2.0], [3.0, 0.0]];
    /// let b = a.non_zero_values();
    /// let expected = HashMap::from([((0, 1), 2.0), ((1, 0), 3.0)]);
    /// assert_eq!(b, expected);
    /// ```
    pub fn non_zero_values(&self) -> HashMap<(usize, usize), f64> {
        let mut res = HashMap::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] != 0.0 {
                    res.insert((i, j), self.data[i][j]);
                }
            }
        }
        res
    }

    /// Validates that the shape of the tensor is the same as the shape of another tensor.
    ///
    /// # Examples
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.validate_shape(&b, "op");
    /// ```
    pub fn validate_shape(&self, other: &Tensor, op: &str) {
        if self.shape() != other.shape() {
            panic!(
                "Tensor.{} invoked with invalid tensor dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
    }

    /// Validates that the columns of the tensor are the same as the rows of another tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::tensor;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.validate_mul_shape(&b, "op");
    /// ```
    pub fn validate_mul_shape(&self, other: &Tensor, op: &str) {
        if self.cols != other.rows {
            panic!(
                "Tensor.{} invoked with invalid tensor dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl From<Vec<Vec<f64>>> for Tensor {
    fn from(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        Tensor { rows, cols, data }
    }
}

impl FromIterator<Tensor1D> for Tensor {
    fn from_iter<I: IntoIterator<Item = Tensor1D>>(iter: I) -> Self {
        let mut data = vec![];
        for row in iter {
            data.push(row);
        }
        let rows = data.len();
        let cols = data[0].len();
        Tensor { rows, cols, data }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = String::new();
        for row in &self.data {
            res.push_str(&format!("{:?}\n", row));
        }
        write!(f, "{}", res)
    }
}

/// Creates a new `Tensor` from a two-dimensional vector of floating point values.
///
/// # Usage
///
/// ```
/// # use engram::tensor;
/// let tensor = tensor![[1.0, 2.0], [3.0, 4.0]];
/// assert_eq!(tensor.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// ```
#[macro_export]
macro_rules! tensor {
    ( $( [ $( $x:expr ),* ] ),* ) => {{
        let mut data = vec![];
        $(
            data.push(vec![$($x as f64),*]);
        )*
        $crate::tensor::Tensor::from(data)
    }};
}
