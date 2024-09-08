use crate::linalg::{Tensor, Tensor1D};

impl Tensor {
    /// Reshapes the tensor to a new shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
    /// let b = tensor![[1.0, 2.0], [4.0, 5.0], [7.0, 8.0], [3.0, 6.0]];
    /// let c = a.reshape(2, 3);
    /// let d = b.reshape(2, 4);
    /// # assert_eq!(c.data, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    /// # assert_eq!(d.data, vec![vec![1.0, 2.0, 4.0, 5.0], vec![7.0, 8.0, 3.0, 6.0]]);
    /// ```
    pub fn reshape(&self, rows: usize, cols: usize) -> Tensor {
        if self.rows * self.cols != rows * cols {
            panic!("New shape must have the same total number of elements as the original tensor.");
        }

        let mut res = Tensor::zeros(rows, cols);
        let mut idx = 0;
        let flat = self.flatten();
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = flat[idx];
                idx += 1;
            }
        }
        res
    }

    /// Returns a resized version of the tensor with the given rows and cols.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let b = a.resize(2, 2);
    /// let c = b.resize(2, 3);
    /// # assert_eq!(b.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(c.data, vec![vec![1.0, 2.0, 0.0], vec![3.0, 4.0, 0.0]]);
    /// ```
    pub fn resize(&self, rows: usize, cols: usize) -> Tensor {
        let mut res = self.clone();
        res.resize_mut(rows, cols);
        res
    }

    /// Returns a resized version of the tensor with the given rows and cols inplace.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let b = a.resize(2, 2);
    /// let c = b.resize(2, 3);
    /// # assert_eq!(b.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(c.data, vec![vec![1.0, 2.0, 0.0], vec![3.0, 4.0, 0.0]]);
    /// ```
    pub fn resize_mut(&mut self, rows: usize, cols: usize) {
        if rows > self.rows || cols > self.cols {
            for row in &mut self.data {
                row.resize(cols, 0.0);
            }
            while self.data.len() < rows {
                self.data.push(vec![0.0; cols]);
            }
        } else {
            self.data.truncate(rows);
            for row in &mut self.data {
                row.truncate(cols);
            }
        }

        self.rows = rows;
        self.cols = cols;
    }

    /// Returns a flattened version of the tensor data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.flatten();
    /// # assert_eq!(b, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn flatten(&self) -> Tensor1D {
        let mut flat_data = Vec::new();
        for row in &self.data {
            flat_data.extend(row);
        }
        Tensor1D::from(flat_data)
    }

    /// Returns a resized version of the tensor with the same shape as the given tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let b = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    /// let c = a.resize_to(&b);
    /// let d = b.resize_to(&a);
    /// # assert_eq!(c.data, vec![vec![1.0, 2.0, 0.0], vec![3.0, 4.0, 0.0], vec![5.0, 6.0, 0.0]]);
    /// # assert_eq!(d.data, vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]]);
    /// ```
    pub fn resize_to(&self, other: &Tensor) -> Tensor {
        self.resize(other.rows, other.cols)
    }

    /// Returns a clipped version of the tensor with values between the given min and max.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[-1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let b = a.clip(0.0, 4.0);
    /// # assert_eq!(b.data, vec![vec![0.0, 2.0], vec![3.0, 4.0], vec![4.0, 4.0]]);
    /// ```
    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        self.map(|x| x.max(min).min(max))
    }

    /// Returns a slice of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let b = a.slice(1, 3);
    /// # assert_eq!(b.data, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
    /// ```
    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        Tensor {
            rows: end - start,
            cols: self.cols,
            data: self.data[start..end].to_vec(),
            grad: None,
        }
    }
}
