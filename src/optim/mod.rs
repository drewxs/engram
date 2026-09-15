//! Optimization algorithms, like `torch.optim`.
//!
//! A training step follows the PyTorch pattern: clear gradients, run the forward pass, call `backward` on the loss and then [`Optimizer::step`].

mod adam;
pub mod lr_scheduler;
mod sgd;

pub use adam::Adam;
pub use sgd::Sgd;

use crate::Tensor;

/// Updates parameters from their accumulated gradients.
pub trait Optimizer {
    /// Applies one update to every parameter that has a gradient.
    fn step(&mut self);

    /// Returns the parameters being optimized.
    fn parameters(&self) -> &[Tensor];

    /// Returns the current learning rate.
    fn learning_rate(&self) -> f32;

    /// Changes the learning rate used by later steps.
    fn set_learning_rate(&mut self, learning_rate: f32);

    /// Clears the gradient of every parameter.
    fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.zero_grad();
        }
    }
}
