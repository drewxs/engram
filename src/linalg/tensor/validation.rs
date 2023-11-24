use crate::Tensor;

impl Tensor {
    /// Validates that the shape of the tensor is the same as the shape of another tensor.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
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
    /// # use engram::*;
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
