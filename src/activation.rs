/// Activation functions and their derivatives
#[derive(Debug)]
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
            Self::Sigmoid => sigmoid(x),
            Self::TanH => tanh(x),
            Self::ReLU => relu(x),
            Self::LeakyReLU => leaky_relu(x),
        }
    }

    /// Returns the derivative of the activation function.
    pub fn gradient(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid(x) * (1.0 - sigmoid(x)),
            Self::TanH => 1.0 - tanh(x).powi(2),
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
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

pub fn leaky_relu(x: f64) -> f64 {
    f64::max(x, 0.01 * x)
}
