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
    /// # assert_eq!(b.data, vec![vec![1.0, 3.0], vec![2.0, 4.0], vec![3.0, 6.0]]);
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
    /// # assert_eq!(a.data, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    /// ```
    pub fn transpose_mut(&mut self) {
        let mut res = Tensor::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        self.rows = res.rows;
        self.cols = res.cols;
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
    /// let c = a.broadcast(&b);
    /// # assert_eq!(b.data, vec![vec![1.0, 2.0, 8.0], vec![3.0, 4.0, 9.0]]);
    /// ```
    pub fn broadcast(&self, other: &Tensor) -> Tensor {
        if self.rows == other.rows && self.cols == other.cols {
            return other.clone();
        }

        if other.rows == 1 && self.cols == other.cols {
            let mut res = Tensor::zeros(self.rows, other.cols);
            for i in 0..self.rows {
                for j in 0..other.cols {
                    res.data[i][j] = other.data[0][j];
                }
            }
            Tensor {
                rows: self.rows,
                cols: other.cols,
                data: res.data,
                grad: None,
            }
        } else if other.cols == 1 && self.rows == other.rows {
            let mut res = Tensor::zeros(self.rows, self.cols);
            for i in 0..other.rows {
                for j in 0..self.cols {
                    res.data[i][j] = other.data[i][0];
                }
            }
            Tensor {
                rows: other.rows,
                cols: self.cols,
                data: res.data,
                grad: None,
            }
        } else {
            other.clone()
        }
    }
}
