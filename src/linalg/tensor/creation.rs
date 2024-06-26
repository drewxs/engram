use crate::{Initializer, Tensor};

impl Tensor {
    /// Creates a new `Tensor` with the specified number of rows, columns, and value.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t1 = Tensor::new(2, 3, 0.0);
    /// let t2 = Tensor::new(3, 2, 0.0);
    /// # assert_eq!(t1.data, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
    /// # assert_eq!(t2.data, vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]]);
    /// ```
    pub fn new(rows: usize, cols: usize, value: f64) -> Tensor {
        Tensor {
            rows,
            cols,
            data: vec![vec![value; cols]; rows],
            grad: None,
        }
    }

    /// Creates a new `Tensor` with the same shape as the provided `Tensor`, filled with the specified value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let t1 = Tensor::new(2, 3, 0.0);
    /// let t2 = Tensor::new_like(&t1, 0.0);
    /// # assert_eq!(t2.data, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
    /// ```
    pub fn new_like(other: &Tensor, value: f64) -> Tensor {
        Tensor::new(other.rows, other.cols, value)
    }

    pub fn zeros(rows: usize, cols: usize) -> Tensor {
        Tensor::new(rows, cols, 0.0)
    }

    pub fn zeros_like(other: &Tensor) -> Tensor {
        Tensor::zeros(other.rows, other.cols)
    }

    pub fn ones(rows: usize, cols: usize) -> Tensor {
        Tensor::new(rows, cols, 1.0)
    }

    pub fn ones_like(other: &Tensor) -> Tensor {
        Tensor::ones(other.rows, other.cols)
    }

    /// Creates a new `Tensor` with the specified size, initialized to the identity matrix.
    ///
    /// # Examples
    /// ```
    /// # use engram::*;
    /// let t = Tensor::identity(3);
    /// # assert_eq!(t.data, vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);
    /// ```
    pub fn identity(size: usize) -> Tensor {
        let mut res = Tensor::zeros(size, size);
        for i in 0..size {
            res.data[i][i] = 1.0;
        }
        res
    }

    /// Creates a new `Tensor` with the specified number of rows and columns, initialized using the provided initializer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let t = Tensor::initialize(2, 3, &Initializer::Xavier);
    /// # assert_eq!(t.data.len(), 2);
    /// # assert_eq!(t.data[0].len(), 3);
    /// # assert!(t.data.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
    /// ```
    pub fn initialize(rows: usize, cols: usize, initializer: &Initializer) -> Tensor {
        let mut res = Tensor::zeros(rows, cols);
        res.data = initializer.initialize(rows, cols);
        res
    }
}
