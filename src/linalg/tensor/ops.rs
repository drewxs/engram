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

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        (&self).add(&other)
    }
}
impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        (&self).add(&other)
    }
}
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        (&self).add(other)
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, other: f64) -> Tensor {
        self.add_scalar(other)
    }
}
impl AddAssign<&Tensor> for &mut Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        self.add_mut(other);
    }
}
impl AddAssign<f64> for &mut Tensor {
    fn add_assign(&mut self, other: f64) {
        self.add_scalar_mut(other);
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        (&self).sub(&other)
    }
}
impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        (&self).sub(&other)
    }
}
impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        (&self).sub(other)
    }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, other: f64) -> Tensor {
        self.sub_scalar(other)
    }
}
impl SubAssign<&Tensor> for &mut Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        self.sub_mut(other);
    }
}
impl SubAssign<f64> for &mut Tensor {
    fn sub_assign(&mut self, other: f64) {
        self.sub_scalar_mut(other);
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        (&self).mul(&other)
    }
}
impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        (&self).mul(&other)
    }
}
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        (&self).mul(other)
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, other: f64) -> Tensor {
        self.mul_scalar(other)
    }
}
impl MulAssign<&Tensor> for &mut Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        self.mul_mut(other);
    }
}
impl MulAssign<f64> for &mut Tensor {
    fn mul_assign(&mut self, other: f64) {
        self.mul_scalar_mut(other);
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        (&self).div(&other)
    }
}
impl Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        (&self).div(&other)
    }
}
impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        (&self).div(other)
    }
}
impl Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, other: f64) -> Tensor {
        self.div_scalar(other)
    }
}
impl DivAssign<&Tensor> for &mut Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        self.div_mut(other);
    }
}
impl DivAssign<f64> for &mut Tensor {
    fn div_assign(&mut self, other: f64) {
        self.div_scalar_mut(other);
    }
}
