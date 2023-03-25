use rand::Rng;

use crate::matrix::Matrix;

pub enum Initializer {
    Xavier,
    Kaiming,
}

impl Initializer {
    pub fn initialize(&self, f_in: usize, f_out: usize) -> Matrix {
        match *self {
            Initializer::Xavier => Self::xavier(f_in, f_out),
            Initializer::Kaiming => Self::kaiming(f_in, f_out),
        }
    }

    // Xavier initialization
    // Use for sigmoid and tanh activation functions
    // Gaussian, µ = 0, σ = √[2 / (f_in + f_out)]
    fn xavier(f_in: usize, f_out: usize) -> Matrix {
        let variance = 2.0 / ((f_in + f_out) as f64);
        let mut rng = rand::thread_rng();
        Matrix::from_shape_fn((f_out, f_in), |_| rng.gen::<f64>() * variance.sqrt())
    }

    // Kaiming initialization
    // Use for ReLU activation functions
    // Gaussian, µ = 0, σ = √[2 / f_in]
    fn kaiming(f_in: usize, f_out: usize) -> Matrix {
        let variance = 2.0 / ((f_in) as f64);
        let mut rng = rand::thread_rng();
        Matrix::from_shape_fn((f_out, f_in), |_| rng.gen::<f64>() * variance.sqrt())
    }
}
