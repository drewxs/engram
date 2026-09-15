#![doc = include_str!("../README.md")]

pub mod autograd;
pub mod data;
pub mod linalg;
pub mod metrics;
pub mod nn;
pub mod optim;
pub mod random;
pub mod safetensors;
pub mod statistics;
pub mod tensor;
pub mod utils;

mod error;
pub mod prelude;

#[cfg(test)]
mod tests;

pub use autograd::no_grad;
pub use error::{Error, Result};
pub use random::manual_seed;
pub use tensor::Tensor;
