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

    /// Returns the derivative of the activation function.
    pub fn gradient(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => Self::sigmoid(x) * (1.0 - Self::sigmoid(x)),
            Self::TanH => 1.0 - Self::tanh(x).powi(2),
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
        }
    }

    /// Returns the result of the sigmoid function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Returns the result of the hyperbolic tangent function.
    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    /// Returns the result of the rectified linear unit function.
    fn relu(x: f64) -> f64 {
        f64::max(x, 0.0)
    }

    /// Returns the result of the leaky rectified linear unit function.
    fn leaky_relu(x: f64) -> f64 {
        f64::max(x, 0.01 * x)
    }
}
