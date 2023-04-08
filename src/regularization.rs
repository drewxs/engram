//! This module provides implementations for L1 and L2 regularization.
//! Regularization can be used to prevent overfitting.
//!
//! # Example
//!
//! ```
//! use engram::{tensor, Tensor, Regularization};
//!
//! let tensor = tensor![[1.0, 2.0], [3.0, 4.0]];
//! let l1_reg = Regularization::L1(0.1);
//! let l2_reg = Regularization::L2(0.1);
//!
//! let l1_loss = l1_reg.loss(&tensor);
//! let l2_loss = l2_reg.loss(&tensor);
//!
//! let l1_grad = l1_reg.grad(&tensor);
//! let l2_grad = l2_reg.grad(&tensor);
//! ```

use crate::tensor::Tensor;

/// An enum representing the regularization methods that can be applied to a tensor.
pub enum Regularization {
    /// L1 regularization with the given lambda value.
    L1(f64),
    /// L2 regularization with the given lambda value.
    L2(f64),
}

impl Regularization {
    /// Calculates the regularization loss for the given tensor based on the current regularization method.
    pub fn loss(&self, tensor: &Tensor) -> f64 {
        match self {
            Regularization::L1(lambda) => {
                *lambda * tensor.data.iter().flatten().map(|x| x.abs()).sum::<f64>()
            }
            Regularization::L2(lambda) => {
                *lambda * tensor.data.iter().flatten().map(|x| x.powi(2)).sum::<f64>()
            }
        }
    }

    /// Calculates the gradient of the regularization term with respect to the given tensor.
    pub fn grad(&self, tensor: &Tensor) -> Tensor {
        match self {
            Regularization::L1(lambda) => {
                let mut grad = Tensor::zeros_like(&tensor);
                for (i, row) in tensor.iter_rows().enumerate() {
                    for (j, val) in row.iter().enumerate() {
                        grad.data[i][j] = val.signum() * *lambda;
                    }
                }
                grad
            }
            Regularization::L2(lambda) => {
                let mut grad = Tensor::zeros_like(&tensor);
                for row in grad.data.iter_mut() {
                    for val in row.iter_mut() {
                        *val = 2.0 * *lambda * *val;
                    }
                }
                grad
            }
        }
    }
}
