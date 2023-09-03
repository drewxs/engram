//! Activation functions and their derivatives

use crate::Tensor;

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
    /// assert_eq!(Activation::Sigmoid.apply(0.0), 0.5, "Sigmoid.apply(0.0)");
    /// assert_eq!(Activation::Sigmoid.apply(1.0), 0.7310585786300049, "Sigmoid.apply(1.0)");
    /// assert_eq!(Activation::TanH.apply(0.0), 0.0, "TanH.apply(0.0)");
    /// assert_eq!(Activation::TanH.apply(1.0), 0.7615941559557649, "TanH.apply(1.0)");
    /// assert_eq!(Activation::ReLU.apply(0.0), 0.0, "ReLU.apply(0.0)");
    /// assert_eq!(Activation::ReLU.apply(1.0), 1.0, "ReLU.apply(1.0)");
    /// assert_eq!(Activation::ReLU.apply(-1.0), 0.0, "ReLU.apply(-1.0)");
    /// assert_eq!(Activation::LeakyReLU.apply(0.0), 0.0, "LeakyReLU.apply(0.0)");
    /// assert_eq!(Activation::LeakyReLU.apply(1.0), 1.0, "LeakyReLU.apply(1.0)");
    /// assert_eq!(Activation::LeakyReLU.apply(-1.0), -0.01, "LeakyReLU.apply(-1.0)");
    /// ```
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => Self::sigmoid(x),
            Self::TanH => Self::tanh(x),
            Self::ReLU => Self::relu(x),
            Self::LeakyReLU => Self::leaky_relu(x),
        }
    }

    /// Returns the derivative of the activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::Activation;
    /// assert_eq!(Activation::Sigmoid.gradient(0.0), 0.25, "Sigmoid.gradient(0.0)");
    /// assert_eq!(Activation::Sigmoid.gradient(1.0), 0.19661193324148185, "Sigmoid.gradient(1.0)");
    /// assert_eq!(Activation::TanH.gradient(0.0), 1.0, "TanH.gradient(0.0)");
    /// assert_eq!(Activation::TanH.gradient(1.0), 0.41997434161402614, "TanH.gradient(1.0)");
    /// assert_eq!(Activation::ReLU.gradient(0.0), 0.0, "ReLU.gradient(0.0)");
    /// assert_eq!(Activation::ReLU.gradient(1.0), 1.0, "ReLU.gradient(1.0)");
    /// assert_eq!(Activation::ReLU.gradient(-1.0), 0.0, "ReLU.gradient(-1.0)");
    /// assert_eq!(Activation::LeakyReLU.gradient(0.0), 0.01, "LeakyReLU.gradient(0.0)");
    /// assert_eq!(Activation::LeakyReLU.gradient(1.0), 1.0, "LeakyReLU.gradient(1.0)");
    /// assert_eq!(Activation::LeakyReLU.gradient(-1.0), 0.01, "LeakyReLU.gradient(-1.0)");
    /// ```
    pub fn gradient(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => Self::d_sigmoid(x),
            Self::TanH => Self::d_tanh(x),
            Self::ReLU => Self::d_relu(x),
            Self::LeakyReLU => Self::d_leaky_relu(x),
        }
    }

    /// Returns a tensor with the activation function applied to each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Activation, tensor};
    /// let tensor = tensor![[1.0, 42.0], [100.0, -4.0]];
    /// let output_sigmoid = Activation::Sigmoid.apply_tensor(&tensor);
    /// let output_tanh = Activation::TanH.apply_tensor(&tensor);
    /// let output_relu = Activation::ReLU.apply_tensor(&tensor);
    /// let output_leaky_relu = Activation::LeakyReLU.apply_tensor(&tensor);
    /// assert_eq!(output_sigmoid, tensor![[0.7310585786300049, 1.0], [1.0, 0.01798620996209156]], "Sigmoid.apply_tensor");
    /// assert_eq!(output_tanh, tensor![[0.7615941559557649, 1.0], [1.0, -0.999329299739067]], "TanH.apply_tensor");
    /// assert_eq!(output_relu, tensor![[1.0, 42.0], [100.0, 0.0]], "ReLU.apply_tensor");
    /// assert_eq!(output_leaky_relu, tensor![[1.0, 42.0], [100.0, -0.04]], "LeakyReLU.apply_tensor");
    /// ```
    pub fn apply_tensor(&self, tensor: &Tensor) -> Tensor {
        tensor.mapv(&|x| self.apply(x))
    }

    /// Returns a tensor with the derivative of the activation function applied to each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::{Activation, tensor};
    /// let tensor = tensor![[1.0, 42.0], [100.0, -4.0]];
    /// let output_sigmoid = Activation::Sigmoid.gradient_tensor(&tensor);
    /// let output_tanh = Activation::TanH.gradient_tensor(&tensor);
    /// let output_relu = Activation::ReLU.gradient_tensor(&tensor);
    /// let output_leaky_relu = Activation::LeakyReLU.gradient_tensor(&tensor);
    /// assert_eq!(output_sigmoid, tensor![[0.19661193324148185, 0.0], [0.0, 0.017662706213291118]], "Sigmoid.gradient_tensor");
    /// assert_eq!(output_tanh, tensor![[0.41997434161402614, 0.0], [0.0, 0.0013409506830258655]], "TanH.gradient_tensor");
    /// assert_eq!(output_relu, tensor![[1.0, 1.0], [1.0, 0.0]], "ReLU.gradient_tensor");
    /// assert_eq!(output_leaky_relu, tensor![[1.0, 1.0], [1.0, 0.01]], "LeakyReLU.gradient_tensor");
    /// ```
    pub fn gradient_tensor(&self, tensor: &Tensor) -> Tensor {
        tensor.mapv(&|x| self.gradient(x))
    }

    /// Returns the result of the sigmoid function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Returns the derivative of the sigmoid function.
    fn d_sigmoid(x: f64) -> f64 {
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }

    /// Returns the result of the hyperbolic tangent function.
    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    /// Returns the derivative of the hyperbolic tangent function.
    fn d_tanh(x: f64) -> f64 {
        1.0 - Self::tanh(x).powi(2)
    }

    /// Returns the result of the rectified linear unit function.
    fn relu(x: f64) -> f64 {
        f64::max(x, 0.0)
    }

    /// Returns the derivative of the rectified linear unit function.
    fn d_relu(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    /// Returns the result of the leaky rectified linear unit function.
    fn leaky_relu(x: f64) -> f64 {
        f64::max(x, 0.01 * x)
    }

    /// Returns the derivative of the leaky rectified linear unit function.
    fn d_leaky_relu(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }
}
