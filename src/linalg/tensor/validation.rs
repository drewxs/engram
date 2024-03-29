use crate::{linalg, Tensor};

impl Tensor {
    pub fn validate(&self, condition: bool, op: &str) {
        if !condition {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{})",
                op, self.rows, self.cols
            );
        }
    }

    pub fn validate_2(&self, other: &Tensor, condition: bool, op: &str) {
        if !condition {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
    }

    pub fn validate_msg(&self, condition: bool, op: &str, msg: &str) {
        if !condition {
            panic!("Tensor.{} invoked with {}", op, msg);
        }
    }

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
        self.validate_2(other, self.is_same_shape(other), op);
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
    /// a.validate_matmul_compatible(&b);
    /// ```
    pub fn validate_matmul_compatible(&self, other: &Tensor) {
        self.validate_2(other, self.is_matmul_compatible(other), "matmul");
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
        self.validate(self.is_square(), op);
    }

    /// Returns true if the tensor is symmetric, i.e. a square matrix with equal elements across the diagonal.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1]];
    /// assert!(t.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] != self.data[j][i] {
                    return false;
                }
            }
        }

        true
    }

    /// Validates that the tensor is symmetric, i.e. a square matrix with equal elements across the diagonal.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1]];
    /// t.validate_symmetric("op");
    /// ```
    pub fn validate_symmetric(&self, op: &str) {
        self.validate_msg(self.is_symmetric(), op, "non-symmetric matrix");
    }

    /// Returns true if the tensor is positive definite, i.e. symmetric and all eigenvalues are positive.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]];
    /// assert!(t.is_positive_definite());
    /// ```
    pub fn is_positive_definite(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        let n = self.rows;

        // Try to compute Cholesky decomposition
        if let Some(chol) = linalg::cholesky(&self) {
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

    /// Validates that the tensor is positive definite, i.e. symmetric and all eigenvalues are positive.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = tensor![[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]];
    /// t.validate_positive_definite("op");
    /// ```
    pub fn validate_positive_definite(&self, op: &str) {
        self.validate_msg(
            self.is_positive_definite(),
            op,
            "non-positive definite matrix",
        );
    }
}
