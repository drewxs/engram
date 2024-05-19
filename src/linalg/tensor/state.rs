use crate::Tensor;

impl Tensor {
    /// Sets the gradient of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.set_grad(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(a.grad, Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]));
    /// ```
    pub fn set_grad(&mut self, grad: Vec<Vec<f64>>) {
        self.grad = Some(grad);
    }

    /// Zeros out the gradient of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.zero_grad();
    /// # assert_eq!(a.grad, Some(vec![vec![0.0, 0.0], vec![0.0, 0.0]]));
    /// ```
    pub fn zero_grad(&mut self) {
        match &mut self.grad {
            Some(grad) => {
                for row in grad.iter_mut() {
                    for val in row.iter_mut() {
                        *val = 0.0;
                    }
                }
            }
            None => {
                self.grad = Some(vec![vec![0.0; self.cols]; self.rows]);
            }
        }
    }
}
