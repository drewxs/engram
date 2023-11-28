use crate::Tensor;

impl Tensor {
    /// Adds two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.add_mut(&b);
    /// assert_eq!(a.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    /// ```
    pub fn add_mut(&mut self, other: &Tensor) {
        self.validate_shape(other, "add_mut");

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
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.add_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
    /// ```
    pub fn add_scalar_mut(&mut self, scalar: f64) {
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
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.sub_mut(&b);
    /// assert_eq!(a.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
    /// ```
    pub fn sub_mut(&mut self, other: &Tensor) {
        self.validate_shape(other, "sub_mut");

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
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.sub_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![-4.0, -3.0], vec![-2.0, -1.0]]);
    /// ```
    pub fn sub_scalar_mut(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] -= scalar;
            }
        }
    }

    /// Multiplies two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.mul_mut(&b);
    /// assert_eq!(a.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
    /// ```
    pub fn mul_mut(&mut self, other: &Tensor) {
        self.validate_mul_shape(other, "mul_mut");

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
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.mul_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    /// ```
    pub fn mul_scalar_mut(&mut self, scalar: f64) {
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
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 8.0], [30.0, 8.0]];
    /// a.div_mut(&b);
    /// assert_eq!(a.data, vec![vec![0.2, 0.25], vec![0.1, 0.5]]);
    /// ```
    pub fn div_mut(&mut self, other: &Tensor) {
        self.validate_shape(other, "div_mut");

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
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.div_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![0.2, 0.4], vec![0.6, 0.8]]);
    /// ```
    pub fn div_scalar_mut(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] /= scalar;
            }
        }
    }
}
