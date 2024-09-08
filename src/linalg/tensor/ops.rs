use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::Tensor;

impl Tensor {
    /// Performs `self + rhs`.
    pub fn add<T: Rhs>(&self, other: T) -> Tensor {
        other.add(self)
    }

    /// Adds two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a + b;
    /// # assert_eq!(c.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    /// ```
    fn add_t(&self, other: &Tensor) -> Tensor {
        self.broadcast_and_apply(other, |a, b| a + b)
    }

    /// Adds two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a += b;
    /// # assert_eq!(a.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    /// ```
    fn add_t_mut(&mut self, other: &Tensor) {
        self.broadcast_and_apply_mut(other, |a, b| a + b);
    }

    /// Adds a scalar to a tensor element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a + 5.0;
    /// # assert_eq!(b.data, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
    /// ```
    fn add_s(&self, scalar: f64) -> Tensor {
        self.apply(|x| x + scalar)
    }

    /// Adds a scalar to a tensor element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a += 5.0;
    /// # assert_eq!(a.data, vec![vec![6.0, 7.0], vec![8.0, 9.0]]);
    /// ```
    fn add_s_mut(&mut self, scalar: f64) {
        self.apply_mut(|x| x + scalar);
    }

    /// Performs `self - rhs`.
    pub fn sub<T: Rhs>(&self, other: T) -> Tensor {
        other.sub(self)
    }

    /// Subtracts two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a - b;
    /// # assert_eq!(c.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
    fn sub_t(&self, other: &Tensor) -> Tensor {
        self.broadcast_and_apply(other, |a, b| a - b)
    }

    /// Subtracts two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a -= b;
    /// # assert_eq!(a.data, vec![vec![-4.0, -4.0], vec![-4.0, -4.0]]);
    /// ```
    fn sub_t_mut(&mut self, other: &Tensor) {
        self.broadcast_and_apply_mut(other, |a, b| a - b)
    }

    /// Subtracts a scalar from a tensor element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a - 5.0;
    /// # assert_eq!(b.data, vec![vec![-4.0, -3.0], vec![-2.0, -1.0]]);
    /// ```
    fn sub_s(&self, scalar: f64) -> Tensor {
        self.apply(|x| x - scalar)
    }

    /// Subtracts a scalar from a tensor element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a -= 5.0;
    /// # assert_eq!(a.data, vec![vec![-4.0, -3.0], vec![-2.0, -1.0]]);
    /// ```
    fn sub_s_mut(&mut self, scalar: f64) {
        self.apply_mut(|x| x - scalar);
    }

    /// Performs `self * rhs`.
    pub fn mul<T: Rhs>(&self, other: T) -> Tensor {
        other.mul(self)
    }

    /// Multiplies two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// let c = a * b;
    /// # assert_eq!(c.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
    /// ```
    fn mul_t(&self, other: &Tensor) -> Tensor {
        self.broadcast_and_apply(other, |a, b| a * b)
    }

    /// Multiplies two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    /// a *= b;
    /// # assert_eq!(a.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
    /// ```
    fn mul_t_mut(&mut self, other: &Tensor) {
        self.broadcast_and_apply_mut(other, |a, b| a * b);
    }

    /// Multiplies a tensor by a scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a * 5.0;
    /// # assert_eq!(b.data, vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    /// ```
    fn mul_s(&self, scalar: f64) -> Tensor {
        self.apply(|x| x * scalar)
    }

    /// Multiplies a tensor by a scalar in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a *= 5.0;
    /// # assert_eq!(a.data, vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    /// ```
    fn mul_s_mut(&mut self, scalar: f64) {
        self.apply_mut(|x| x * scalar);
    }

    /// Performs `self / rhs`.
    pub fn div<T: Rhs>(&self, other: T) -> Tensor {
        other.div(self)
    }

    /// Divides two tensors element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 8.0], [30.0, 8.0]];
    /// let c = a / b;
    /// # assert_eq!(c.data, vec![vec![0.2, 0.25], vec![0.1, 0.5]]);
    /// ```
    fn div_t(&self, other: &Tensor) -> Tensor {
        self.broadcast_and_apply(other, |a, b| a / b)
    }

    /// Divides two tensors element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[5.0, 8.0], [30.0, 8.0]];
    /// a /= b;
    /// # assert_eq!(a.data, vec![vec![0.2, 0.25], vec![0.1, 0.5]]);
    /// ```
    fn div_t_mut(&mut self, other: &Tensor) {
        self.broadcast_and_apply_mut(other, |a, b| a / b);
    }

    /// Divides a tensor by a scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = a / 5.0;
    /// # assert_eq!(b.data, vec![vec![0.2, 0.4], vec![0.6, 0.8]]);
    /// ```
    fn div_s(&self, scalar: f64) -> Tensor {
        self.apply(|x| x / scalar)
    }

    /// Divides a tensor by a scalar in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// a /= 5.0;
    /// # assert_eq!(a.data, vec![vec![0.2, 0.4], vec![0.6, 0.8]]);
    /// ```
    fn div_s_mut(&mut self, scalar: f64) {
        self.apply_mut(|x| x / scalar);
    }

    fn apply<F>(&self, op: F) -> Tensor
    where
        F: Fn(f64) -> f64,
    {
        let mut res = self.clone();
        res.apply_mut(op);
        res
    }

    fn apply_mut<F>(&mut self, op: F)
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = op(self.data[i][j]);
            }
        }
    }

    /// Broadcasts two tensors and applies a function element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let t1 = tensor![[1.0], [2.0]];
    /// let t2 = tensor![3.0, 4.0];
    /// let t3 = t1.broadcast_and_apply(&t2, |a, b| a + b);
    /// # assert_eq!(t3.data, vec![vec![4.0, 5.0], vec![5.0, 6.0]]);
    /// ```
    pub fn broadcast_and_apply<F>(&self, other: &Tensor, op: F) -> Tensor
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut res = self.clone();
        res.broadcast_and_apply_mut(other, op);
        res
    }

    /// Broadcasts two tensors and applies a function element-wise in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut t1 = tensor![[1.0], [2.0]];
    /// let t2 = tensor![3.0, 4.0];
    /// t1.broadcast_and_apply_mut(&t2, |a, b| a + b);
    /// # assert_eq!(t1.data, vec![vec![4.0, 5.0], vec![5.0, 6.0]]);
    /// ```
    pub fn broadcast_and_apply_mut<F>(&mut self, other: &Tensor, op: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        self.validate_broadcast_compatible(other);

        let new_rows = self.rows.max(other.rows);
        let new_cols = self.cols.max(other.cols);

        let mut result = Tensor::zeros(new_rows, new_cols);

        for i in 0..new_rows {
            for j in 0..new_cols {
                let self_value = if self.rows == 1 {
                    self.data[0][j % self.cols]
                } else if self.cols == 1 {
                    self.data[i % self.rows][0]
                } else {
                    self.data[i][j]
                };

                let other_value = if other.rows == 1 {
                    other.data[0][j % other.cols]
                } else if other.cols == 1 {
                    other.data[i % other.rows][0]
                } else {
                    other.data[i][j]
                };

                result.data[i][j] = op(self_value, other_value);
            }
        }

        *self = result;
    }
}

pub trait Rhs {
    fn add(&self, lhs: &Tensor) -> Tensor;
    fn sub(&self, lhs: &Tensor) -> Tensor;
    fn mul(&self, lhs: &Tensor) -> Tensor;
    fn div(&self, lhs: &Tensor) -> Tensor;
}

impl Rhs for &Tensor {
    fn add(&self, lhs: &Tensor) -> Tensor {
        lhs.add_t(self)
    }
    fn sub(&self, lhs: &Tensor) -> Tensor {
        lhs.sub_t(self)
    }
    fn mul(&self, lhs: &Tensor) -> Tensor {
        lhs.mul_t(self)
    }
    fn div(&self, lhs: &Tensor) -> Tensor {
        lhs.div_t(self)
    }
}

impl Rhs for f64 {
    fn add(&self, lhs: &Tensor) -> Tensor {
        lhs.add_s(*self)
    }
    fn sub(&self, lhs: &Tensor) -> Tensor {
        lhs.sub_s(*self)
    }
    fn mul(&self, lhs: &Tensor) -> Tensor {
        lhs.mul_s(*self)
    }
    fn div(&self, lhs: &Tensor) -> Tensor {
        lhs.div_s(*self)
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        self.add_t(&rhs)
    }
}
impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.add_t(rhs)
    }
}
impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        self.add_t(&rhs)
    }
}
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.add_t(rhs)
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor {
        self.add_s(rhs)
    }
}
impl Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor {
        self.add_s(rhs)
    }
}
impl AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        self.add_t_mut(&rhs);
    }
}
impl AddAssign<f64> for Tensor {
    fn add_assign(&mut self, rhs: f64) {
        self.add_s_mut(rhs);
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self.sub_t(&rhs)
    }
}
impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.sub_t(rhs)
    }
}
impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self.sub_t(&rhs)
    }
}
impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.sub_t(rhs)
    }
}
impl Sub<f64> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Tensor {
        self.sub_s(rhs)
    }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Tensor {
        self.sub_s(rhs)
    }
}
impl SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: Tensor) {
        self.sub_t_mut(&rhs);
    }
}
impl SubAssign<f64> for Tensor {
    fn sub_assign(&mut self, rhs: f64) {
        self.sub_s_mut(rhs);
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        self.mul_t(&rhs)
    }
}
impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.mul_t(rhs)
    }
}
impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        self.mul_t(&rhs)
    }
}
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.mul_t(rhs)
    }
}
impl Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor {
        self.mul_s(rhs)
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor {
        self.mul_s(rhs)
    }
}
impl MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: Tensor) {
        self.mul_t_mut(&rhs);
    }
}
impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, rhs: f64) {
        self.mul_s_mut(rhs);
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self.div_t(&rhs)
    }
}
impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.div_t(rhs)
    }
}
impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self.div_t(&rhs)
    }
}
impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.div_t(rhs)
    }
}
impl Div<f64> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor {
        self.div_s(rhs)
    }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor {
        self.div_s(rhs)
    }
}
impl DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, rhs: Tensor) {
        self.div_t_mut(&rhs);
    }
}
impl DivAssign<f64> for Tensor {
    fn div_assign(&mut self, rhs: f64) {
        self.div_s_mut(rhs);
    }
}
