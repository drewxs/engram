//! Create and manipulate neural network layers.
//!
//! This module provides a Layer struct for representing a single layer in a neural network,
//! along with methods for feeding inputs through the layer and performing backpropagation.

mod dense;
mod layer;

pub use dense::DenseLayer;
pub use layer::Layer;
