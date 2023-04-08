use rand::Rng;

use crate::tensor::Tensor2D;

#[derive(Debug)]
pub enum Initializer {
    Xavier,
    Kaiming,
}

impl Initializer {
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
        let mut data = Vec::with_capacity(f_out);
        for _ in 0..f_out {
            let mut row = Vec::with_capacity(f_in);
            for _ in 0..f_in {
                row.push(rng.gen::<f64>() * std_dev);
            }
            data.push(row);
        }
        data
    }
}
