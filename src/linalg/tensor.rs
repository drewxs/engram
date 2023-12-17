//! Tensors representing two-dimensional matrices of floating point values.
//!
//! Also provides type aliases for one- and two-dimensional vectors of floating point values,
//! as well as methods for initializing, manipulating, and performing mathematical operations on tensors.

mod access;
mod computation;
mod creation;
mod decomposition;
mod display;
mod eq;
mod from;
mod iter;
mod macros;
mod manipulation;
mod ops;
mod sparse;
mod transformation;
mod validation;

/// A one-dimensional matrix of floating point values.
pub type Tensor1D = Vec<f64>;
/// A two-dimensional matrix of floating point values.
pub type Tensor2D = Vec<Tensor1D>;

/// A matrix of floating point values, represented as a two-dimensional vector.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The number of rows in the matrix.
    pub rows: usize,
    /// The number of columns in the matrix.
    pub cols: usize,
    /// The data in the matrix, represented as a two-dimensional vector.
    pub data: Tensor2D,
}
