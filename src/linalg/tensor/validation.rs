use crate::{linalg, Tensor};

impl Tensor {
    pub(super) fn is_same_shape(&self, other: &Tensor) -> bool {
        self.shape() == other.shape()
    }

    pub(super) fn validate_same_shape(&self, other: &Tensor, op: &str) {
        self.validate(other, self.is_same_shape(other), op);
    }

    pub(super) fn is_matmul_compatible(&self, other: &Tensor) -> bool {
        self.cols == other.rows
    }

    pub(super) fn validate_matmul_compatible(&self, other: &Tensor) {
        self.validate(other, self.is_matmul_compatible(other), "matmul");
    }

    pub(super) fn is_broadcast_compatible(&self, other: &Tensor) -> bool {
        (self.rows == other.rows || self.rows == 1 || other.rows == 1)
            && (self.cols == other.cols || self.cols == 1 || other.cols == 1)
    }

    pub(super) fn validate_broadcast_compatible(&self, other: &Tensor) {
        self.validate(other, self.is_broadcast_compatible(other), "broadcast");
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
        if self.rows != self.cols {
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

    /// Returns true if the tensor is positive definite, i.e. symmetric and all eigenvalues are positive.
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

    fn validate(&self, other: &Tensor, condition: bool, op: &str) {
        if !condition {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
    }
}
