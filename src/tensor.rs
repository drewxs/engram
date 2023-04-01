use crate::initializer::Initializer;

/// A one-dimensional matrix of floating point values.
pub type Tensor1D = Vec<f64>;
/// A two-dimensional matrix of floating point values.
pub type Tensor2D = Vec<Tensor1D>;

/// A matrix of floating point values, represented as a two-dimensional vector.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The number of rows in the matrix.
    pub rows: usize,
    /// The number of columns in the matrix.
    pub cols: usize,
    /// The data in the matrix, represented as a two-dimensional vector.
    pub data: Tensor2D,
}

impl Tensor {
    /// Creates a new `Tensor` of zeros with the specified number of rows and columns.
    pub fn zeros(rows: usize, cols: usize) -> Tensor {
        Tensor {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    /// Creates a new `Tensor` from a two-dimensional vector of floating point values.
    pub fn from(data: Tensor2D) -> Tensor {
        Tensor {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    /// Creates a new `Tensor` with the specified number of rows and columns, initialized using the provided initializer.
    pub fn initialize(rows: usize, cols: usize, initializer: &Initializer) -> Tensor {
        let mut res = Tensor::zeros(rows, cols);

        res.data = initializer.initialize(rows, cols);

        res
    }

    /// Performs matrix multiplication between two tensors.
    pub fn mul(&mut self, other: &Tensor) -> Tensor {
        if self.cols != other.rows {
            panic!("Tensor.mul invoked with invalid tensor dimensions");
        }

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

    /// Adds two tensors element-wise.
    pub fn add(&mut self, other: &Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.add invoked with invalid tensor dimensions");
        }

        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        res
    }

    /// Multiplies two tensors element-wise.
    pub fn dot(&mut self, other: &Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.dot invoked with invalid tensor dimensions");
        }

        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        res
    }

    /// Subtracts two tensors element-wise.
    pub fn sub(&mut self, other: &Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.sub invoked with invalid tensor dimensions");
        }

        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        res
    }

    /// Applies a function to each element in the tensor.
    pub fn map(&mut self, function: &dyn Fn(f64) -> f64) -> Tensor {
        let data = (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| function(x)).collect())
            .collect::<Tensor2D>();

        Tensor {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Returns the transpose of the tensor.
    pub fn transpose(&mut self) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        res
    }
}

/// Creates a new `Tensor` from a two-dimensional vector of floating point values.
#[macro_export]
macro_rules! tensor {
    ( $( [ $( $x:expr ),* ] ),* ) => {{
        let mut data = vec![];
        $(
            data.push(vec![$($x as f64),*]);
        )*
        data
    }};
}

pub use tensor;
