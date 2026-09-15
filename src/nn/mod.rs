//! Neural network layers, losses and initialization.

mod activation;
mod dropout;
pub mod functional;
pub mod init;
mod linear;
mod sequential;
pub mod utils;

pub use activation::{Gelu, LeakyRelu, LogSoftmax, Relu, Sigmoid, Softmax, Tanh};
pub use dropout::Dropout;
pub use linear::Linear;
pub use sequential::Sequential;

use crate::Tensor;

/// A layer or model that maps an input tensor to an output tensor, like `torch.nn.Module`.
///
/// Layers with weights also implement [`Module::parameters`], and layers that behave differently during training, such as dropout, implement [`Module::set_training`]. Any `Fn(&Tensor) -> Tensor` is a module too, which makes one-off transformations easy to drop into a [`Sequential`].
///
/// ```
/// # use engram::{nn::{Linear, Module}, Tensor};
/// struct Mlp {
///     hidden: Linear,
///     output: Linear,
/// }
///
/// impl Module for Mlp {
///     fn forward(&self, input: &Tensor) -> Tensor {
///         self.output.forward(&self.hidden.forward(input).relu())
///     }
///
///     fn parameters(&self) -> Vec<Tensor> {
///         [self.hidden.parameters(), self.output.parameters()].concat()
///     }
/// }
///
/// let model = Mlp { hidden: Linear::new(4, 16), output: Linear::new(16, 2) };
/// assert_eq!(model.forward(&Tensor::randn([8, 4])).shape(), &[8, 2]);
/// assert_eq!(model.num_parameters(), 4 * 16 + 16 + 16 * 2 + 2);
/// ```
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns handles to every trainable tensor. They share data with the module, so updating them updates the module.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    /// Switches between training and evaluation behavior.
    fn set_training(&mut self, _training: bool) {}

    fn train(&mut self) {
        self.set_training(true);
    }

    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Clears the gradient of every parameter.
    fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.zero_grad();
        }
    }

    /// Counts the trainable elements across all parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(Tensor::numel).sum()
    }
}

impl<F: Fn(&Tensor) -> Tensor> Module for F {
    fn forward(&self, input: &Tensor) -> Tensor {
        self(input)
    }
}
