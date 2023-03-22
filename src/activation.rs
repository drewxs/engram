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
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

pub enum Activation {
    Sigmoid,
    TanH,
    ReLU,
    LeakyReLU,
}

impl Default for Activation {
    fn default() -> Self {
        Self::Sigmoid
    }
}
