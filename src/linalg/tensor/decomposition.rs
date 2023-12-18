use crate::Tensor;

impl Tensor {
    /// Cholesky decomposition of a symmetric, positive-definite matrix.
    /// Returns the product of the lower triangular matrix and its conjugate transpose.
    /// Returns None if the input matrix is not not square or positive-definite.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t1 = tensor![[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]];
    /// let t2 = t1.cholesky().unwrap();
    /// assert_eq!(t2, tensor![[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]);
    /// ```
    pub fn cholesky(&self) -> Option<Self> {
        if !self.is_square() {
            return None;
        }

        let mut result = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..(i + 1) {
                let mut sum = 0.0;

                if j == i {
                    for k in 0..j {
                        sum += result.data[j][k] * result.data[j][k];
                    }
                    result.data[j][j] = (self.data[j][j] - sum).sqrt();
                } else {
                    for k in 0..j {
                        sum += result.data[i][k] * result.data[j][k];
                    }
                    if result.data[j][j] > f64::EPSILON {
                        result.data[i][j] = (self.data[i][j] - sum) / result.data[j][j];
                    } else {
                        return None;
                    }
                }
            }
        }

        Some(result)
    }
}
