//! Activation functions and their derivatives

mod leaky_relu;
mod relu;
mod sigmoid;
mod tanh;

pub use leaky_relu::*;
pub use relu::*;
pub use sigmoid::*;
pub use tanh::*;

/// An activation function.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Activation {
    Sigmoid,
    TanH,
    ReLU,
    LeakyReLU,
}

impl Activation {
    /// Returns the result of the activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::Activation;
    /// Activation::Sigmoid.apply(1.0);
    /// ```
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid(x),
            Self::TanH => tanh(x),
            Self::ReLU => relu(x),
            Self::LeakyReLU => leaky_relu(x),
        }
    }

    /// Returns the derivative of the activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::Activation;
    /// Activation::ReLU.grad(1.0);
    /// ```
    pub fn grad(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => d_sigmoid(x),
            Self::TanH => d_tanh(x),
            Self::ReLU => d_relu(x),
            Self::LeakyReLU => d_leaky_relu(x),
        }
    }
}
