use crate::{linalg::Tensor2D, Activation, Tensor};

impl Tensor {
    /// Performs matrix multiplication between two tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let b = tensor![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    /// let c = a.matmul(&b);
    /// assert_eq!(c.data, vec![vec![58.0, 64.0], vec![139.0, 154.0]]);
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        self.validate_matmul_compatible(other);

        let mut res = Tensor::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                res.data[i][j] = sum;
            }
        }
        res
    }

    /// Applies a function to each element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.mapv(&|x| x * 2.0);
    /// assert_eq!(b.data, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    /// ```
    pub fn mapv(&self, function: &dyn Fn(f64) -> f64) -> Tensor {
        let data = (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| function(x)).collect())
            .collect::<Tensor2D>();

        Tensor {
            rows: self.rows,
            cols: self.cols,
            data,
            grad: self.grad.clone(),
        }
    }

    /// Applies a function to each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.mapv_mut(&|x| x * 2.0);
    /// assert_eq!(a.data, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    /// ```
    pub fn mapv_mut(&mut self, function: &dyn Fn(f64) -> f64) {
        self.data = (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| function(x)).collect())
            .collect::<Tensor2D>();
    }

    /// Returns the square of each element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.square();
    /// assert_eq!(b.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn square(&self) -> Tensor {
        self.mapv(&|x| x * x)
    }

    /// Squares each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.square_mut();
    /// assert_eq!(a.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn square_mut(&mut self) {
        self.mapv_mut(&|x| x * x);
    }

    /// Returns the square root of each element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 4.0], [9.0, 16.0]];
    /// let b = a.sqrt();
    /// assert_eq!(b.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// ```
    pub fn sqrt(&self) -> Tensor {
        self.mapv(&|x| x.sqrt())
    }

    /// Takes the square root of each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 4.0], [9.0, 16.0]];
    /// a.sqrt_mut();
    /// assert_eq!(a.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// ```
    pub fn sqrt_mut(&mut self) {
        self.mapv_mut(&|x| x.sqrt());
    }

    /// Returns each element in the tensor raised to the given exponent.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.pow(2.0);
    /// assert_eq!(b.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn pow(&self, exponent: f64) -> Tensor {
        self.mapv(&|x| x.powf(exponent))
    }

    /// Raises each element in the tensor to the given exponent in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.pow_mut(2.0);
    /// assert_eq!(a.data, vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
    /// ```
    pub fn pow_mut(&mut self, exponent: f64) {
        self.mapv_mut(&|x| x.powf(exponent));
    }

    /// Returns each element in the tensor applied with the natural logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.ln();
    /// assert_eq!(b.data, vec![vec![0.0, 0.6931471805599453], vec![1.0986122886681098, 1.3862943611198906]]);
    /// ```
    pub fn ln(&self) -> Tensor {
        self.mapv(&|x| x.ln())
    }

    /// Applies the natural logarithm to each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.ln_mut();
    /// assert_eq!(a.data, vec![vec![0.0, 0.6931471805599453], vec![1.0986122886681098, 1.3862943611198906]]);
    /// ```
    pub fn ln_mut(&mut self) {
        self.mapv_mut(&|x| x.ln());
    }

    /// Returns each element in the tensor applied with the base 2 logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [4.0, 8.0]];
    /// let b = a.log2();
    /// assert_eq!(b.data, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    /// ```
    pub fn log2(&self) -> Tensor {
        self.mapv(&|x| x.log2())
    }

    /// Applies the base 2 logarithm to each element in the tensor in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [4.0, 8.0]];
    /// a.log2_mut();
    /// assert_eq!(a.data, vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    /// ```
    pub fn log2_mut(&mut self) {
        self.mapv_mut(&|x| x.log2());
    }

    /// Returns a tensor with the absolute value of each element in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[-1.0, 2.0], [-3.0, 4.0]];
    /// let b = a.abs();
    /// assert_eq!(b.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// ```
    pub fn abs(&self) -> Tensor {
        self.mapv(&|x| x.abs())
    }

    /// Computes the dot product between two tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0], [2.0], [3.0]];
    /// let b = tensor![[4.0], [5.0], [6.0]];
    /// let c = a.dot(&b);
    /// assert_eq!(c, 32.0);
    /// ```
    pub fn dot(&self, other: &Tensor) -> f64 {
        self.validate_same_shape(other, "dot");

        let flat_self = self.flatten();
        let flat_other = other.flatten();
        let mut res = 0.0;
        for i in 0..flat_self.len() {
            res += flat_self[i] * flat_other[i];
        }
        res
    }

    /// Returns the sum of all elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.sum();
    /// assert_eq!(b, 10.0);
    /// ```
    pub fn sum(&self) -> f64 {
        self.data.iter().flatten().fold(0.0, |acc, x| acc + x)
    }

    /// Returns the sum of all elements in the tensor along the given axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let sum_cols = a.sum_axis(0);
    /// let sum_rows = a.sum_axis(1);
    /// assert_eq!(sum_cols.data, vec![vec![9.0, 12.0]]);
    /// assert_eq!(sum_rows.data, vec![vec![3.0], vec![7.0], vec![11.0]]);
    /// ```
    pub fn sum_axis(&self, axis: u8) -> Tensor {
        if axis == 0 {
            let mut res = Tensor::zeros(1, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    res.data[0][j] += self.data[i][j];
                }
            }
            res
        } else if axis == 1 {
            let mut res = Tensor::zeros(self.rows, 1);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    res.data[i][0] += self.data[i][j];
                }
            }
            res
        } else {
            panic!("Invalid axis value: {}", axis);
        }
    }

    /// Returns the mean of all elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.mean();
    /// assert_eq!(b, 2.5);
    /// ```
    pub fn mean(&self) -> f64 {
        self.sum() / (self.rows * self.cols) as f64
    }

    /// Returns the p-norm of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.norm(2.0);
    /// assert_eq!(b, 5.477225575051661);
    /// ```
    pub fn norm(&self, p: f64) -> f64 {
        self.mapv(&|x| x.powf(p)).sum().sqrt()
    }

    /// Returns a new tensor with each element in the tensor activated by the given activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.activate(&Activation::Sigmoid);
    /// assert_eq!(b.data, vec![vec![0.7310585786300049, 0.8807970779778823], vec![0.9525741268224334, 0.9820137900379085]]);
    /// ```
    pub fn activate(&self, activation: &Activation) -> Tensor {
        self.mapv(&|x| activation.apply(x))
    }

    /// Mutates the tensor in-place with each element in the tensor activated by the given activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.activate_mut(&Activation::Sigmoid);
    /// assert_eq!(a.data, vec![vec![0.7310585786300049, 0.8807970779778823], vec![0.9525741268224334, 0.9820137900379085]]);
    /// ```
    pub fn activate_mut(&mut self, activation: &Activation) {
        self.mapv_mut(&|x| activation.apply(x));
    }

    /// Returns a new tensor with each element in the tensor activated by the derivative of the given activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.grad(&Activation::Sigmoid);
    /// assert_eq!(b.data, vec![vec![0.19661193324148185, 0.10499358540350662], vec![0.045176659730912, 0.017662706213291107]]);
    /// ```
    pub fn grad(&mut self, activation: &Activation) -> Tensor {
        let res = self.mapv(&|x| activation.grad(x));
        self.set_grad(res.data.clone());
        res
    }
}
