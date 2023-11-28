use std::collections::HashMap;

use crate::Tensor;

impl Tensor {
    /// Performs sparse matrix multiplication between two tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]];
    /// let b = tensor![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    /// let c = a.sparse_matmul(&b);
    /// assert_eq!(c.data, vec![vec![7.0, 8.0], vec![45.0, 50.0], vec![22.0, 24.0]]);
    /// ```
    pub fn sparse_matmul(&self, other: &Tensor) -> Tensor {
        self.validate_mul_shape(other, "matmul");

        let sparse_self = self.sparse();
        let sparse_other = other.sparse();

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

    /// Returns a vector of the indices of all non-zero elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use engram::*;
    /// let a = tensor![[0.0, 2.0], [3.0, 0.0]];
    /// let b = a.sparse();
    /// let expected = HashMap::from([((0, 1), 2.0), ((1, 0), 3.0)]);
    /// assert_eq!(b, expected);
    /// ```
    pub fn sparse(&self) -> HashMap<(usize, usize), f64> {
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
}
