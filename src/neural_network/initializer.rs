//! Initializations for weight matrices.

use rand::Rng;

use crate::Tensor2D;

#[derive(Debug)]
pub enum Initializer {
    Xavier,
    Kaiming,
}

impl Initializer {
    /// Initializes a 2D tensor with random values based on the specified initialization technique.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let xavier = Initializer::Xavier.initialize(2, 3);
    /// let kaiming = Initializer::Kaiming.initialize(5, 3);
    /// assert_eq!(xavier.len(), 2);
    /// assert_eq!(xavier[0].len(), 3);
    /// assert!(xavier.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
    /// assert_eq!(kaiming.len(), 5);
    /// assert_eq!(kaiming[0].len(), 3);
    /// assert!(kaiming.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
    /// ```
    pub fn initialize(&self, f_in: usize, f_out: usize) -> Tensor2D {
        match *self {
            Initializer::Xavier => Self::xavier(f_in, f_out),
            Initializer::Kaiming => Self::kaiming(f_in, f_out),
        }
    }

    /// Xavier/Glorot initialization
    /// Use for sigmoid and tanh activation functions
    /// Gaussian, µ = 0, σ = √[2 / (f_in + f_out)]
    fn xavier(f_in: usize, f_out: usize) -> Tensor2D {
        Self::initialize_data(f_in, f_out, 2.0 / ((f_in + f_out) as f64).sqrt())
    }

    /// Kaiming initialization
    /// Use for ReLU activation functions
    /// Gaussian, µ = 0, σ = √[2 / f_in]
    fn kaiming(f_in: usize, f_out: usize) -> Tensor2D {
        Self::initialize_data(f_in, f_out, 2.0 / ((f_in) as f64).sqrt())
    }

    /// Initializes a 2D tensor with random values based on the specified standard deviation.
    fn initialize_data(f_in: usize, f_out: usize, std_dev: f64) -> Tensor2D {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(f_in);
        for _ in 0..f_in {
            let mut row = Vec::with_capacity(f_out);
            for _ in 0..f_out {
                row.push(rng.gen::<f64>() * std_dev);
            }
            data.push(row);
        }
        data
    }
}
