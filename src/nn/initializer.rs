//! Initializations for weight matrices.

use rand::Rng;

use crate::linalg::Tensor2D;

/// Initialization methods for weight matrices.
#[derive(Debug, PartialEq)]
pub enum Initializer {
    /// Xavier (Glorot) initialization.
    /// Gaussian, µ = 0, σ = √[2 / (f_in + f_out)]
    Xavier,
    /// Kaiming (He) initialization.
    /// Gaussian, µ = 0, σ = √[2 / f_in]
    Kaiming,
    /// Constant initialization.
    /// All weights are set to the given value.
    Constant(f64),
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
    /// # assert_eq!(xavier.len(), 2);
    /// # assert_eq!(xavier[0].len(), 3);
    /// # assert!(xavier.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
    /// # assert_eq!(kaiming.len(), 5);
    /// # assert_eq!(kaiming[0].len(), 3);
    /// # assert!(kaiming.iter().all(|w| w.iter().all(|x| x.abs() <= 1.0)));
    /// ```
    pub fn initialize(&self, f_in: usize, f_out: usize) -> Tensor2D {
        match *self {
            Initializer::Xavier => Self::xavier(f_in, f_out),
            Initializer::Kaiming => Self::kaiming(f_in, f_out),
            Initializer::Constant(val) => Self::constant(f_in, f_out, val),
        }
    }

    fn xavier(f_in: usize, f_out: usize) -> Tensor2D {
        Self::initialize_data(f_in, f_out, 2.0 / ((f_in + f_out) as f64).sqrt())
    }

    fn kaiming(f_in: usize, f_out: usize) -> Tensor2D {
        Self::initialize_data(f_in, f_out, 2.0 / ((f_in) as f64).sqrt())
    }

    fn constant(f_in: usize, f_out: usize, val: f64) -> Tensor2D {
        Self::initialize_with_closure(f_in, f_out, &mut || val)
    }

    /// Initializes a 2D tensor with random values based on the specified standard deviation.
    fn initialize_data(f_in: usize, f_out: usize, std_dev: f64) -> Tensor2D {
        let mut rng = rand::thread_rng();
        Self::initialize_with_closure(f_in, f_out, &mut || rng.gen::<f64>() * std_dev)
    }

    fn initialize_with_closure(
        f_in: usize,
        f_out: usize,
        closure: &mut dyn FnMut() -> f64,
    ) -> Tensor2D {
        let mut data = Vec::with_capacity(f_in);
        for _ in 0..f_in {
            let mut row = Vec::with_capacity(f_out);
            for _ in 0..f_out {
                row.push(closure());
            }
            data.push(row);
        }
        data
    }
}
