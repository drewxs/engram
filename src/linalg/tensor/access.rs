use crate::Tensor;

impl Tensor {
    /// Returns the shape of the tensor as a tuple of (rows, columns).
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let tensor = Tensor::zeros(2, 3);
    /// let tensor_b = Tensor::zeros(3, 2);
    /// assert_eq!(tensor.shape(), (2, 3));
    /// assert_eq!(tensor_b.shape(), (3, 2));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let tensor = Tensor::zeros(2, 3);
    /// let tensor_b = Tensor::zeros(5, 4);
    /// assert_eq!(tensor.size(), 6);
    /// assert_eq!(tensor_b.size(), 20);
    /// ```
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    /// Returns the first element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let tensor = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// assert_eq!(tensor.first(), 1.0);
    /// ```
    pub fn first(&self) -> f64 {
        self.data[0][0]
    }
}
