#![doc = include_str!("../README.md")]

pub mod autograd;
pub mod data;
pub mod linalg;
pub mod metrics;
pub mod nn;
pub mod optim;
pub mod random;
pub mod statistics;
pub mod tensor;
pub mod utils;

pub mod prelude;

#[cfg(test)]
mod tests;

pub use autograd::no_grad;
pub use random::manual_seed;
pub use tensor::Tensor;
