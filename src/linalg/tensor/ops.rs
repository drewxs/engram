use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::Tensor;

impl Tensor {
    /// Adds two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a.add(&b);
    /// assert_eq!(c.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    /// ```
    pub fn add(&self, other: &Tensor) -> Tensor {
        let mut res = self.clone();
        res.add_mut(other);
        res
    }

    /// Adds two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.add_mut(&b);
    /// assert_eq!(a.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    /// ```
    pub fn add_mut(&mut self, other: &Tensor) {
        for i in 0..self.rows {
            if i >= other.rows {
                break;
            }
            for j in 0..self.cols {
                if j >= other.cols {
                    break;
                }
                self.data[i][j] += other.data[i][j];
            }
        }
    }

    /// Adds a scalar to a tensor element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.add_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
    /// ```
    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        let mut res = self.clone();
        res.add_scalar_mut(scalar);
        res
    }

    /// Adds a scalar to a tensor element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.add_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
    /// ```
    pub fn add_scalar_mut(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] += scalar;
            }
        }
    }

    /// Subtracts two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a.sub(&b);
    /// assert_eq!(c.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
    pub fn sub(&self, other: &Tensor) -> Tensor {
        let mut res = self.clone();
        res.sub_mut(other);
        res
    }

    /// Subtracts two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.sub_mut(&b);
    /// assert_eq!(a.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
    /// ```
    pub fn sub_mut(&mut self, other: &Tensor) {
        for i in 0..self.rows {
            if i >= other.rows {
                break;
            }
            for j in 0..self.cols {
                if j >= other.cols {
                    break;
                }
                self.data[i][j] -= other.data[i][j];
            }
        }
    }

    /// Subtracts a scalar from a tensor element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.sub_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![-4.0, -3.0], vec![-2.0, -1.0]]);
    /// ```
    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        let mut res = self.clone();
        res.sub_scalar_mut(scalar);
        res
    }

    /// Subtracts a scalar from a tensor element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.sub_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![-4.0, -3.0], vec![-2.0, -1.0]]);
    /// ```
    pub fn sub_scalar_mut(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] -= scalar;
            }
        }
    }

    /// Multiplies two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a.mul(&b);
    /// assert_eq!(c.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
    /// ```
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let mut res = self.clone();
        res.mul_mut(other);
        res
    }

    /// Multiplies two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a.mul_mut(&b);
    /// assert_eq!(a.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
    /// ```
    pub fn mul_mut(&mut self, other: &Tensor) {
        for i in 0..self.rows {
            if i >= other.rows {
                break;
            }
            for j in 0..self.cols {
                if j >= other.cols {
                    break;
                }
                self.data[i][j] *= other.data[i][j];
            }
        }
    }

    /// Multiplies a tensor by a scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.mul_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    /// ```
    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        let mut res = self.clone();
        res.mul_scalar_mut(scalar);
        res
    }

    /// Multiplies a tensor by a scalar in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.mul_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    /// ```
    pub fn mul_scalar_mut(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] *= scalar;
            }
        }
    }

    /// Divides two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 8.0], [30.0, 8.0]];
    /// let c = a.div(&b);
    /// assert_eq!(c.data, vec![vec![0.2, 0.25], vec![0.1, 0.5]]);
    /// ```
    pub fn div(&self, other: &Tensor) -> Tensor {
        let mut res = self.clone();
        res.div_mut(other);
        res
    }

    /// Divides two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 8.0], [30.0, 8.0]];
    /// a.div_mut(&b);
    /// assert_eq!(a.data, vec![vec![0.2, 0.25], vec![0.1, 0.5]]);
    /// ```
    pub fn div_mut(&mut self, other: &Tensor) {
        for i in 0..self.rows {
            if i >= other.rows {
                break;
            }
            for j in 0..self.cols {
                if j >= other.cols {
                    break;
                }
                self.data[i][j] /= other.data[i][j];
            }
        }
    }

    /// Divides a tensor by a scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a.div_scalar(5.0);
    /// assert_eq!(b.data, vec![vec![0.2, 0.4], vec![0.6, 0.8]]);
    /// ```
    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        let mut res = self.clone();
        res.div_scalar_mut(scalar);
        res
    }

    /// Divides a tensor by a scalar in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a.div_scalar_mut(5.0);
    /// assert_eq!(a.data, vec![vec![0.2, 0.4], vec![0.6, 0.8]]);
    /// ```
    pub fn div_scalar_mut(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] /= scalar;
            }
        }
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.add(rhs)
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor {
        self.add_scalar(rhs)
    }
}
impl Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor {
        self.add_scalar(rhs)
    }
}
impl AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        self.add_mut(&rhs);
    }
}
impl AddAssign<f64> for Tensor {
    fn add_assign(&mut self, rhs: f64) {
        self.add_scalar_mut(rhs);
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.sub(rhs)
    }
}
impl Sub<f64> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Tensor {
        self.sub_scalar(rhs)
    }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Tensor {
        self.sub_scalar(rhs)
    }
}
impl SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: Tensor) {
        self.sub_mut(&rhs);
    }
}
impl SubAssign<f64> for Tensor {
    fn sub_assign(&mut self, rhs: f64) {
        self.sub_scalar_mut(rhs);
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.mul(rhs)
    }
}
impl Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor {
        self.mul_scalar(rhs)
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor {
        self.mul_scalar(rhs)
    }
}
impl MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: Tensor) {
        self.mul_mut(&rhs);
    }
}
impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, rhs: f64) {
        self.mul_scalar_mut(rhs);
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.div(rhs)
    }
}
impl Div<f64> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor {
        self.div_scalar(rhs)
    }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor {
        self.div_scalar(rhs)
    }
}
impl DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, rhs: Tensor) {
        self.div_mut(&rhs);
    }
}
impl DivAssign<f64> for Tensor {
    fn div_assign(&mut self, rhs: f64) {
        self.div_scalar_mut(rhs);
    }
}
