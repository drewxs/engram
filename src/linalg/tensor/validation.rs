use crate::Tensor;

impl Tensor {
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
        if self.shape() != other.shape() {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
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
        if self.cols != other.rows {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{} and {}x{})",
                op, self.rows, self.cols, other.rows, other.cols
            );
        }
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
        if self.rows != self.cols {
            panic!(
                "Tensor.{} invoked with invalid dimensions ({}x{})",
                op, self.rows, self.cols
            );
        }
    }
}
