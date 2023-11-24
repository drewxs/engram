use crate::{Tensor, Tensor1D};

impl Tensor {
    /// Returns an iterator over the rows of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut tensor = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let mut iter = tensor.iter_rows();
    /// assert_eq!(iter.next(), Some(&vec![1.0, 2.0, 3.0]));
    /// assert_eq!(iter.next(), Some(&vec![4.0, 5.0, 6.0]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_rows(&self) -> impl Iterator<Item = &Tensor1D> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the rows of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut tensor = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let mut iter = tensor.iter_rows_mut();
    /// assert_eq!(iter.next(), Some(&mut vec![1.0, 2.0, 3.0]));
    /// assert_eq!(iter.next(), Some(&mut vec![4.0, 5.0, 6.0]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut Tensor1D> {
        self.data.iter_mut()
    }
}

impl FromIterator<Tensor1D> for Tensor {
    fn from_iter<I: IntoIterator<Item = Tensor1D>>(iter: I) -> Self {
        let mut data = vec![];
        for row in iter {
            data.push(row);
        }
        let rows = data.len();
        let cols = data[0].len();
        Tensor { rows, cols, data }
    }
}
