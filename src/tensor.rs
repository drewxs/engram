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

    /// Creates a new `Tensor` of zeros with the same shape as the provided `Tensor`.
    pub fn zeros_like(other: &Tensor) -> Tensor {
        Tensor::zeros(other.rows, other.cols)
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

    /// Returns the shape of the tensor as a tuple of (rows, columns).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    /// Returns an iterator over the rows of the tensor.
    pub fn iter_rows(&self) -> impl Iterator<Item = &Tensor1D> {
        self.data.iter()
    }

    /// Adds two tensors element-wise.
    pub fn add(&self, other: &Tensor) -> Tensor {
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

    /// Adds two tensors element-wise in-place.
    pub fn add_assign(&mut self, other: &Tensor) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.add_assign invoked with invalid tensor dimensions");
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] += other.data[i][j];
            }
        }
    }

    /// Adds a scalar to a tensor element-wise.
    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + scalar;
            }
        }

        res
    }

    /// Adds a scalar to a tensor element-wise in-place.
    pub fn add_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] += scalar;
            }
        }
    }

    /// Subtracts two tensors element-wise.
    pub fn sub(&self, other: &Tensor) -> Tensor {
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

    /// Subtracts two tensors element-wise in-place.
    pub fn sub_assign(&mut self, other: &Tensor) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.sub_assign invoked with invalid tensor dimensions");
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] -= other.data[i][j];
            }
        }
    }

    /// Subtracts a scalar from a tensor element-wise.
    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - scalar;
            }
        }

        res
    }

    /// Subtracts a scalar from a tensor element-wise in-place.
    pub fn sub_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] -= scalar;
            }
        }
    }

    /// Performs matrix multiplication between two tensors.
    pub fn mul(&self, other: &Tensor) -> Tensor {
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

    /// Performs matrix multiplication between two tensors in-place.
    pub fn mul_assign(&mut self, other: &Tensor) {
        if self.cols != other.rows {
            panic!("Tensor.mul_assign invoked with invalid tensor dimensions");
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

        self.data = res.data;
    }

    /// Multiplies a tensor by a scalar.
    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * scalar;
            }
        }

        res
    }

    /// Multiplies a tensor by a scalar in-place.
    pub fn mul_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] *= scalar;
            }
        }
    }

    /// Divides two tensors element-wise.
    pub fn div(&self, other: &Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.div invoked with invalid tensor dimensions");
        }

        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] / other.data[i][j];
            }
        }

        res
    }

    /// Divides two tensors element-wise in-place.
    pub fn div_assign(&mut self, other: &Tensor) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.div_assign invoked with invalid tensor dimensions");
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] /= other.data[i][j];
            }
        }
    }

    /// Divides a tensor by a scalar.
    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] / scalar;
            }
        }

        res
    }

    /// Divides a tensor by a scalar in-place.
    pub fn div_scalar_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] /= scalar;
            }
        }
    }

    /// Multiplies two tensors element-wise.
    pub fn dot(&self, other: &Tensor) -> Tensor {
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

    /// Multiplies two tensors element-wise in-place.
    pub fn dot_assign(&mut self, other: &Tensor) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Tensor.dot_assign invoked with invalid tensor dimensions");
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] *= other.data[i][j];
            }
        }
    }

    /// Applies a function to each element in the tensor.
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
        }
    }

    /// Applies a function to each element in the tensor in-place.
    pub fn mapv_assign(&mut self, function: &dyn Fn(f64) -> f64) {
        self.data = (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| function(x)).collect())
            .collect::<Tensor2D>();
    }

    /// Returns the square of each element in the tensor.
    pub fn square(&self) -> Tensor {
        self.mapv(&|x| x * x)
    }

    /// Squares each element in the tensor in-place.
    pub fn square_assign(&mut self) {
        self.mapv_assign(&|x| x * x);
    }

    /// Returns the square root of each element in the tensor.
    pub fn sqrt(&self) -> Tensor {
        self.mapv(&|x| x.sqrt())
    }

    /// Takes the square root of each element in the tensor in-place.
    pub fn sqrt_assign(&mut self) {
        self.mapv_assign(&|x| x.sqrt());
    }

    /// Returns each element in the tensor raised to the given exponent.
    pub fn pow(&self, exponent: f64) -> Tensor {
        self.mapv(&|x| x.powf(exponent))
    }

    /// Raises each element in the tensor to the given exponent in-place.
    pub fn pow_assign(&mut self, exponent: f64) {
        self.mapv_assign(&|x| x.powf(exponent));
    }

    /// Returns each element in the tensor applied with the natural logarithm.
    pub fn ln(&self) -> Tensor {
        self.mapv(&|x| x.ln())
    }

    /// Applies the natural logarithm to each element in the tensor in-place.
    pub fn ln_assign(&mut self) {
        self.mapv_assign(&|x| x.ln());
    }

    /// Returns each element in the tensor applied with the base 2 logarithm.
    pub fn log2(&self) -> Tensor {
        self.mapv(&|x| x.log2())
    }

    /// Applies the base 2 logarithm to each element in the tensor in-place.
    pub fn log2_assign(&mut self) {
        self.mapv_assign(&|x| x.log2());
    }

    /// Returns the sum of all elements in the tensor.
    pub fn sum(&self) -> f64 {
        self.data
            .clone()
            .into_iter()
            .flatten()
            .fold(0.0, |acc, x| acc + x)
    }

    /// Returns the mean of all elements in the tensor.
    pub fn mean(&self) -> f64 {
        self.sum() / (self.rows * self.cols) as f64
    }

    /// Returns the transpose of the tensor.
    pub fn transpose(&self) -> Tensor {
        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        res
    }

    /// Transposes the tensor in-place.
    pub fn transpose_assign(&mut self) {
        let mut res = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        self.data = res.data;
    }

    /// Returns a slice of the tensor.
    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        let mut res = Tensor::zeros(1, end - start);

        for i in 0..(end - start) {
            res.data[0][i] = self.data[0][start + i];
        }

        res
    }

    /// Returns the first element in the tensor.
    pub fn first(&self) -> f64 {
        self.data[0][0]
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
        crate::tensor::Tensor::from(data)
    }};
}

pub use tensor;
