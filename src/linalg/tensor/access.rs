use crate::Tensor;

impl Tensor {
    /// Returns the shape of the tensor as a tuple of (rows, columns).
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let t1 = Tensor::zeros(2, 3);
    /// let t2 = Tensor::zeros(3, 2);
    /// t1.shape();
    /// t2.shape();
    /// # assert_eq!(t1.shape(), (2, 3));
    /// # assert_eq!(t2.shape(), (3, 2));
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
    /// let t1 = Tensor::zeros(2, 3);
    /// let t2 = Tensor::zeros(5, 4);
    /// t1.size();
    /// t2.size();
    /// # assert_eq!(t1.size(), 6);
    /// # assert_eq!(t2.size(), 20);
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
    /// let t = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// t.first();
    /// # assert_eq!(t.first(), 1.0);
    /// ```
    pub fn first(&self) -> f64 {
        self.data[0][0]
    }
}
