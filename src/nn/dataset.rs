//! Dataset
//!
//! This is used to create datasets from input and target tensors,
//! and to create batches from those datasets.
//!
//! Used in the `fit` method of `NeuralNetwork` to train on a dataset.
//!
//! # Examples
//!
//! ```
//! # use engram::*;
//! let inputs = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let targets = tensor![[0.0], [1.0], [0.0]];
//!
//! let dataset = Dataset::new(inputs, targets);
//! let loss_fn = Loss::MSE;
//! let epochs = 1000;
//! let batch_size = 4;
//!
//! let mut nn = NeuralNetwork::default();
//! nn.fit(&dataset, &loss_fn, epochs, batch_size);
//! ```

use crate::Tensor;

pub struct Dataset {
    pub inputs: Tensor,
    pub targets: Tensor,
}

impl Dataset {
    /// Create a new dataset from the given input and target tensors.
    /// The number of samples in the input and target tensors must be the same.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let inputs = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let targets = tensor![[0.0], [1.0], [0.0]];
    /// let dataset = Dataset::new(inputs, targets);
    /// assert_eq!(dataset.inputs.data, &[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    /// ```
    pub fn new(inputs: Tensor, targets: Tensor) -> Self {
        assert_eq!(
            inputs.rows, targets.rows,
            "Inputs and targets must have the same number of samples"
        );
        Dataset { inputs, targets }
    }

    /// Create batches of input and target tensors with the given batch size.
    /// The last batch may be smaller if the number of samples is not divisible by the batch size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let inputs = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let targets = tensor![[0.0], [1.0], [0.0]];
    /// let dataset = Dataset::new(inputs, targets);
    /// let batches = dataset.batches(2);
    /// assert_eq!(batches.len(), 2);
    /// assert_eq!(batches[0].0, tensor![[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(batches[0].1, tensor![[0.0], [1.0]]);
    /// ```
    pub fn batches(&self, batch_size: usize) -> Vec<(Tensor, Tensor)> {
        let mut batches = Vec::new();
        let samples = self.inputs.rows;

        for start in (0..samples).step_by(batch_size) {
            let end = (start + batch_size).min(samples);
            let input_batch = self.inputs.slice(start, end);
            let target_batch = self.targets.slice(start, end);

            batches.push((input_batch, target_batch));
        }

        batches
    }
}
