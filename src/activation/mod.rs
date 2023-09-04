//! Activation functions and their derivatives

pub mod leaky_relu;
pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub use leaky_relu::*;
pub use relu::*;
pub use sigmoid::*;
pub use tanh::*;

/// An activation function.
#[derive(Clone, Copy, Debug)]
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
    /// Activation::Sigmoid.apply(0.0);
    /// Activation::TanH.apply(0.0);
    /// Activation::ReLU.apply(0.0);
    /// Activation::LeakyReLU.apply(0.0);
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
    /// Activation::Sigmoid.gradient(0.0);
    /// Activation::TanH.gradient(0.0);
    /// Activation::ReLU.gradient(0.0);
    /// Activation::LeakyReLU.gradient(0.0);
    /// ```
    pub fn gradient(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => d_sigmoid(x),
            Self::TanH => d_tanh(x),
            Self::ReLU => d_relu(x),
            Self::LeakyReLU => d_leaky_relu(x),
        }
    }
}
