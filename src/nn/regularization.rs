//! L1/L2 regularization.
//!
//! Regularization can be used to prevent overfitting.
//!
//! # Example
//!
//! ```
//! use engram::{tensor, nn::{Regularization, L1, L2}};
//!
//! let tensor = tensor![[1.0, 2.0], [3.0, 4.0]];
//! let l1_reg = L1(0.1);
//! let l2_reg = L2(0.1);
//!
//! let l1_loss = l1_reg.loss(&tensor);
//! let l2_loss = l2_reg.loss(&tensor);
//!
//! let l1_grad = l1_reg.grad(&tensor);
//! let l2_grad = l2_reg.grad(&tensor);
//! ```

use std::fmt::Debug;

use crate::Tensor;

pub trait Regularization: Debug {
    fn loss(&self, tensor: &Tensor) -> f64;
    fn grad(&self, tensor: &Tensor) -> Tensor;
}

/// L1 regularization with the given lambda value.
/// Also known as LASSO, used to encourage sparsity.
#[derive(Clone, Debug)]
pub struct L1(pub f64);

impl Regularization for L1 {
    fn loss(&self, tensor: &Tensor) -> f64 {
        self.0 * tensor.data.iter().flatten().map(|x| x.abs()).sum::<f64>()
    }

    fn grad(&self, tensor: &Tensor) -> Tensor {
        let mut grad = Tensor::zeros_like(tensor);
        for (i, row) in tensor.iter_rows().enumerate() {
            for (j, val) in row.iter().enumerate() {
                grad.data[i][j] = val.signum() * self.0;
            }
        }
        grad
    }
}

/// L2 regularization with the given lambda value.
/// Also known as Ridge, used to encourage small weights.
#[derive(Clone, Debug)]
pub struct L2(pub f64);

impl Regularization for L2 {
    fn loss(&self, tensor: &Tensor) -> f64 {
        self.0 * tensor.data.iter().flatten().map(|x| x.powi(2)).sum::<f64>()
    }

    fn grad(&self, tensor: &Tensor) -> Tensor {
        let mut grad = Tensor::zeros_like(tensor);
        for row in grad.data.iter_mut() {
            for val in row.iter_mut() {
                *val *= 2.0 * self.0
            }
        }
        grad
    }
}
