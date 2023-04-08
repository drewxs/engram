use crate::Tensor;

/// Activation functions and their derivatives
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Sigmoid,
    TanH,
    ReLU,
    LeakyReLU,
}

impl Activation {
    /// Returns the result of the activation function.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => Self::sigmoid(x),
            Self::TanH => Self::tanh(x),
            Self::ReLU => Self::relu(x),
            Self::LeakyReLU => Self::leaky_relu(x),
        }
    }

    /// Returns a tensor with the activation function applied to each element.
    pub fn apply_tensor(&self, tensor: &Tensor) -> Tensor {
        tensor.mapv(&|x| self.apply(x))
    }

    /// Returns the derivative of the activation function.
    pub fn gradient(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => Self::sigmoid_derivative(x),
            Self::TanH => 1.0 - Self::tanh_derivative(x),
            Self::ReLU => Self::relu_derivative(x),
            Self::LeakyReLU => Self::leaky_relu_derivative(x),
        }
    }

    /// Returns a tensor with the derivative of the activation function applied to each element.
    pub fn gradient_tensor(&self, tensor: &Tensor) -> Tensor {
        tensor.mapv(&|x| self.gradient(x))
    }

    /// Returns the result of the sigmoid function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Returns the derivative of the sigmoid function.
    fn sigmoid_derivative(x: f64) -> f64 {
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }

    /// Returns the result of the hyperbolic tangent function.
    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    /// Returns the derivative of the hyperbolic tangent function.
    fn tanh_derivative(x: f64) -> f64 {
        1.0 - Self::tanh(x).powi(2)
    }

    /// Returns the result of the rectified linear unit function.
    fn relu(x: f64) -> f64 {
        f64::max(x, 0.0)
    }

    /// Returns the derivative of the rectified linear unit function.
    fn relu_derivative(x: f64) -> f64 {
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
    fn leaky_relu_derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }
}
