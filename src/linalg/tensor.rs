//! Tensors representing two-dimensional matrices of floating point values.
//!
//! Also provides type aliases for one- and two-dimensional vectors of floating point values,
//! as well as methods for initializing, manipulating, and performing mathematical operations on tensors.

mod access;
mod computation;
mod creation;
mod display;
mod eq;
mod from;
mod iter;
mod macros;
mod manipulation;
mod ops;
mod sparse;
mod state;
mod transformation;
mod validation;

/// A one-dimensional tensor of floating point values.
pub type Tensor1D = Vec<f64>;
/// A two-dimensional tensor of floating point values.
pub type Tensor2D = Vec<Tensor1D>;

/// A tensor of floating point values.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The number of rows in the tensor.
    pub rows: usize,
    /// The number of columns in the tensor.
    pub cols: usize,
    /// The data in the tensor, represented as a two-dimensional vector.
    pub data: Tensor2D,
    /// Gradient of the tensor.
    pub grad: Option<Tensor2D>,
}
