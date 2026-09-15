//! Neural network layers, losses and initialization.

mod activation;
mod conv;
mod dropout;
mod embedding;
mod flatten;
pub mod functional;
pub mod init;
mod linear;
mod normalization;
mod pooling;
mod sequential;
pub mod utils;

pub use activation::{Gelu, LeakyRelu, LogSoftmax, Relu, Sigmoid, Softmax, Tanh};
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::Flatten;
pub use linear::Linear;
pub use normalization::LayerNorm;
pub use pooling::{AvgPool2d, MaxPool2d};
pub use sequential::Sequential;

use std::collections::BTreeMap;

use crate::{Error, Result, Tensor};

/// Tensors keyed by parameter or buffer name, as returned by [`Module::state_dict`].
pub type StateDict = BTreeMap<String, Tensor>;

/// A layer or model that maps an input tensor to an output tensor, like `torch.nn.Module`.
///
/// Layers with weights implement [`Module::named_parameters`], which also drives [`Module::parameters`], [`Module::state_dict`] and [`Module::load_state_dict`]. Layers that behave differently during training, such as dropout, implement [`Module::set_training`]. Any `Fn(&Tensor) -> Tensor` is a module too, which makes one-off transformations easy to drop into a [`Sequential`].
///
/// ```
/// # use engram::{nn::{prefixed, Linear, Module}, Tensor};
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
///     fn named_parameters(&self) -> Vec<(String, Tensor)> {
///         [
///             prefixed("hidden", self.hidden.named_parameters()),
///             prefixed("output", self.output.named_parameters()),
///         ]
///         .concat()
///     }
/// }
///
/// let model = Mlp { hidden: Linear::new(4, 16), output: Linear::new(16, 2) };
/// assert_eq!(model.forward(&Tensor::randn([8, 4])).shape(), &[8, 2]);
/// assert_eq!(model.num_parameters(), 4 * 16 + 16 + 16 * 2 + 2);
/// assert!(model.state_dict().contains_key("output.bias"));
/// ```
pub trait Module {
    /// Computes the output of the module for `input`.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns every trainable tensor with a dotted name, such as `0.weight` for the first layer of a [`Sequential`]. The tensors share data with the module, so updating them updates the module.
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// Returns state that is saved with the module but not trained, such as running statistics.
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// Returns every trainable tensor in the order of [`Module::named_parameters`].
    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(_, parameter)| parameter)
            .collect()
    }

    /// Switches between training and evaluation behavior.
    fn set_training(&mut self, _training: bool) {}

    /// Puts the module in training mode.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Puts the module in evaluation mode, which turns off training behavior such as dropout.
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

    /// Returns every parameter and buffer by name, like `state_dict` in PyTorch. The tensors are snapshots that later training does not change.
    fn state_dict(&self) -> StateDict {
        self.named_parameters()
            .into_iter()
            .chain(self.named_buffers())
            .map(|(name, tensor)| (name, tensor.detach()))
            .collect()
    }

    /// Copies the tensors in `state` into the module's parameters and buffers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::StateDict`] and leaves the module unchanged when a name is missing, `state` holds a name the module does not have, or a shape differs.
    fn load_state_dict(&self, state: &StateDict) -> Result<()> {
        let targets: Vec<(String, Tensor)> = self
            .named_parameters()
            .into_iter()
            .chain(self.named_buffers())
            .collect();
        for (name, target) in &targets {
            let source = state
                .get(name)
                .ok_or_else(|| Error::StateDict(format!("missing {name}")))?;
            if source.shape() != target.shape() {
                return Err(Error::StateDict(format!(
                    "{name} has shape {:?} but the module expects {:?}",
                    source.shape(),
                    target.shape()
                )));
            }
        }
        if let Some(name) = state
            .keys()
            .find(|name| targets.iter().all(|(n, _)| n != *name))
        {
            return Err(Error::StateDict(format!("unexpected {name}")));
        }
        for (name, target) in &targets {
            target.copy_from(&state[name]);
        }
        Ok(())
    }
}

impl<F: Fn(&Tensor) -> Tensor> Module for F {
    fn forward(&self, input: &Tensor) -> Tensor {
        self(input)
    }
}

/// Puts `prefix.` in front of every name, for building the named parameters of a module from its children.
pub fn prefixed(prefix: &str, entries: Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
    entries
        .into_iter()
        .map(|(name, tensor)| (format!("{prefix}.{name}"), tensor))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Sequential {
        Sequential::new()
            .add(Linear::new(3, 4))
            .add(Relu)
            .add(Linear::without_bias(4, 2))
    }

    #[test]
    fn names_parameters_by_position() {
        let names: Vec<String> = model()
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, ["0.weight", "0.bias", "2.weight"]);
    }

    #[test]
    fn state_dict_round_trips_between_models() {
        let (source, target) = (model(), model());
        let x = Tensor::randn([5, 3]);
        assert_ne!(source.forward(&x), target.forward(&x));
        target.load_state_dict(&source.state_dict()).unwrap();
        assert_eq!(source.forward(&x), target.forward(&x));
    }

    #[test]
    fn state_dict_is_a_snapshot() {
        let layer = Linear::new(2, 2);
        let state = layer.state_dict();
        crate::nn::init::zeros(&layer.weight);
        assert_ne!(state["weight"], layer.weight);
    }

    #[test]
    fn rejects_mismatched_state() {
        let target = model();
        let mut state = model().state_dict();
        state.insert("extra".to_string(), Tensor::zeros([1]));
        assert!(
            matches!(target.load_state_dict(&state), Err(Error::StateDict(m)) if m == "unexpected extra")
        );

        let mut state = model().state_dict();
        state.remove("0.bias");
        assert!(target.load_state_dict(&state).is_err());

        let mut state = model().state_dict();
        state.insert("2.weight".to_string(), Tensor::zeros([2, 5]));
        let before = target.state_dict();
        assert!(target.load_state_dict(&state).is_err());
        assert_eq!(target.state_dict(), before);
    }
}
