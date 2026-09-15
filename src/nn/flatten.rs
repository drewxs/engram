use crate::{nn::Module, Tensor};

/// Flattens a range of dimensions into one, like `torch.nn.Flatten`. The default keeps the batch dimension and merges the rest.
///
/// ```
/// # use engram::{nn::{Flatten, Module}, Tensor};
/// assert_eq!(Flatten::default().forward(&Tensor::zeros([8, 3, 4, 4])).shape(), &[8, 48]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Flatten {
    start_dim: isize,
    end_dim: isize,
}

impl Flatten {
    /// Creates a layer that merges dimensions `start_dim` through `end_dim`, inclusive.
    pub fn new(start_dim: isize, end_dim: isize) -> Flatten {
        Flatten { start_dim, end_dim }
    }
}

impl Default for Flatten {
    fn default() -> Flatten {
        Flatten::new(1, -1)
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.flatten(self.start_dim, self.end_dim)
    }
}
