use crate::Tensor;

impl Tensor {
    /// Returns the transpose of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
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
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.transpose_mut();
    /// assert_eq!(a.data, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    /// ```
    pub fn transpose_mut(&mut self) {
        let mut res = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        self.data = res.data;
    }

    /// Broadcasts the tensor to another tensor's shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]];
    /// let c = a.broadcast_to(&b);
    /// assert_eq!(b.data, vec![vec![1.0, 2.0, 8.0], vec![3.0, 4.0, 9.0]]);
    /// ```
    pub fn broadcast_to(&self, other: &Tensor) -> Tensor {
        let mut res = Tensor::zeros(other.rows, other.cols);
        for i in 0..other.rows {
            for j in 0..other.cols {
                res.data[i][j] = self.data[i % self.rows][j % self.cols];
            }
        }
        res
    }
}
