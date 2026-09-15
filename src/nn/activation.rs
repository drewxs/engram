//! Activation layers.

use crate::{nn::Module, Tensor};

macro_rules! activation {
    ($(#[$doc:meta])* $name:ident => $method:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.$method()
            }
        }
    };
}

activation!(
    /// Rectified linear unit, `max(0, x)`.
    Relu => relu
);

activation!(
    /// Logistic sigmoid.
    Sigmoid => sigmoid
);

activation!(
    /// Hyperbolic tangent.
    Tanh => tanh
);

activation!(
    /// Gaussian error linear unit.
    Gelu => gelu
);

/// Leaky rectified linear unit with the given negative slope.
#[derive(Clone, Copy, Debug)]
pub struct LeakyRelu(pub f32);

impl Default for LeakyRelu {
    fn default() -> Self {
        LeakyRelu(0.01)
    }
}

impl Module for LeakyRelu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.leaky_relu(self.0)
    }
}

/// Softmax over the given dimension.
#[derive(Clone, Copy, Debug)]
pub struct Softmax(pub isize);

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.softmax(self.0)
    }
}

/// Log-softmax over the given dimension.
#[derive(Clone, Copy, Debug)]
pub struct LogSoftmax(pub isize);

impl Module for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.log_softmax(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    #[test]
    fn layers_delegate_to_tensor_ops() {
        let x = tensor![[-1, 0.5]];
        assert_eq!(Relu.forward(&x), x.relu());
        assert_eq!(Sigmoid.forward(&x), x.sigmoid());
        assert_eq!(Tanh.forward(&x), x.tanh());
        assert_eq!(Gelu.forward(&x), x.gelu());
        assert_eq!(LeakyRelu::default().forward(&x), x.leaky_relu(0.01));
        assert_eq!(Softmax(-1).forward(&x), x.softmax(-1));
        assert_eq!(LogSoftmax(1).forward(&x), x.log_softmax(1));
    }
}
