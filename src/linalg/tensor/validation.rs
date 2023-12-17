use crate::Tensor;

impl Tensor {
    /// Returns true if the tensor is the same shape as another tensor.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t1 = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let t2 = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// assert!(t1.is_same_shape(&t2));
    /// ```
    pub fn is_same_shape(&self, other: &Tensor) -> bool {
        self.shape() == other.shape()
    }

    /// Validates that the shape of the tensor is the same as the shape of another tensor.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t1 = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let t2 = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// t1.validate_same_shape(&t2, "op");
    /// ```
    pub fn validate_same_shape(&self, other: &Tensor, op: &str) {
        if !self.is_same_shape(other) {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
    }

    /// Returns true if columns of the tensor are the same as the rows of another tensor.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// assert!(a.is_matmul_compatible(&b));
    /// ```
    pub fn is_matmul_compatible(&self, other: &Tensor) -> bool {
        self.cols == other.rows
    }

    /// Validates that the columns of the tensor are the same as the rows of another tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.validate_matmul_compatible(&b, "op");
    /// ```
    pub fn validate_matmul_compatible(&self, other: &Tensor, op: &str) {
        if !self.is_matmul_compatible(other) {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
    }

    /// Returns true if the tensor is square.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// assert!(t.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Validates that the tensor is square.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let t = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// t.validate_square("op");
    /// ```
    pub fn validate_square(&self, op: &str) {
        if !self.is_square() {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{})",
                op, self.rows, self.cols
            );
        }
    }

    /// Validates that the tensor is symmetric.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1]];
    /// t.validate_symmetric("op");
    /// ```
    pub fn validate_symmetric(&self, op: &str) {
        self.validate_square(op);

        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] != self.data[j][i] {
                    panic!("Tensor.{} invoked with non-symmetric matrix", op);
                }
            }
        }
    }

    /// Returns true if the tensor is positive definite.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]];
    /// assert!(t.is_positive_definite());
    /// ```
    pub fn is_positive_definite(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }

        let n = self.rows;

        // Try to compute Cholesky decomposition
        if let Some(chol) = self.cholesky() {
            // Check if the diagonal elements of Cholesky matrix are positive
            for i in 0..n {
                if chol.data[i][i] <= 0.0 {
                    return false;
                }
            }
            return true;
        }

        false
    }

    /// Validates that the tensor is positive definite.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]];
    /// t.validate_positive_definite("op");
    /// ```
    pub fn validate_positive_definite(&self, op: &str) {
        if !self.is_positive_definite() {
            panic!("Tensor.{} invoked with non-positive definite matrix", op);
        }
    }
}
