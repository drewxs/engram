//! Datasets and batching, like `torch.utils.data`.

mod loader;

pub use loader::{Batches, DataLoader};

use crate::Tensor;

/// An indexable collection of samples.
pub trait Dataset {
    /// The type of one sample, which is also the type of a collated batch.
    type Item;

    /// Returns the number of samples.
    fn len(&self) -> usize;

    /// Loads the sample at `index`.
    fn get(&self, index: usize) -> Self::Item;

    /// Returns true if the dataset holds no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Loads the samples at `indices` as one batch. The default gathers them one at a time and merges them with [`Collate`], and datasets backed by tensors override it to gather rows in one pass.
    fn get_batch(&self, indices: &[usize]) -> Self::Item
    where
        Self::Item: Collate,
    {
        Collate::collate(indices.iter().map(|&i| self.get(i)).collect())
    }
}

impl<D: Dataset + ?Sized> Dataset for &D {
    type Item = D::Item;

    fn len(&self) -> usize {
        (**self).len()
    }

    fn get(&self, index: usize) -> D::Item {
        (**self).get(index)
    }

    fn get_batch(&self, indices: &[usize]) -> D::Item
    where
        D::Item: Collate,
    {
        (**self).get_batch(indices)
    }
}

/// Merges individual samples into a batch.
pub trait Collate: Sized {
    /// Combines samples into one batch.
    fn collate(items: Vec<Self>) -> Self;
}

/// Stacks samples along a new leading dimension.
impl Collate for Tensor {
    fn collate(items: Vec<Tensor>) -> Tensor {
        Tensor::stack(&items, 0)
    }
}

impl<A: Collate, B: Collate> Collate for (A, B) {
    fn collate(items: Vec<(A, B)>) -> (A, B) {
        let (a, b): (Vec<A>, Vec<B>) = items.into_iter().unzip();
        (A::collate(a), B::collate(b))
    }
}

/// Pairs inputs with targets along their first dimension, like `torch.utils.data.TensorDataset`.
#[derive(Clone, Debug)]
pub struct TensorDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    /// # Panics
    ///
    /// Panics if the tensors hold a different number of samples.
    pub fn new(inputs: Tensor, targets: Tensor) -> TensorDataset {
        assert!(
            inputs.ndim() > 0 && targets.ndim() > 0 && inputs.shape()[0] == targets.shape()[0],
            "TensorDataset: inputs {:?} and targets {:?} must hold the same number of samples",
            inputs.shape(),
            targets.shape()
        );
        TensorDataset { inputs, targets }
    }

    /// Returns the inputs, one sample per entry along the first dimension.
    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    /// Returns the targets, aligned with the inputs along the first dimension.
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }
}

impl Dataset for TensorDataset {
    type Item = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    fn get(&self, index: usize) -> (Tensor, Tensor) {
        (self.inputs.select(0, index), self.targets.select(0, index))
    }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        (
            self.inputs.index_select(0, indices),
            self.targets.index_select(0, indices),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    struct Squares;

    impl Dataset for Squares {
        type Item = (Tensor, Tensor);

        fn len(&self) -> usize {
            5
        }

        fn get(&self, index: usize) -> (Tensor, Tensor) {
            let x = index as f32;
            (tensor![x], tensor![x * x])
        }
    }

    #[test]
    fn default_batches_collate_samples() {
        let (x, y) = Squares.get_batch(&[1, 3]);
        assert_eq!(x, tensor![[1], [3]]);
        assert_eq!(y, tensor![[1], [9]]);
    }

    #[test]
    fn tensor_dataset_gathers_rows() {
        let dataset = TensorDataset::new(tensor![[1, 2], [3, 4], [5, 6]], tensor![0, 1, 0]);
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(1), (tensor![3, 4], Tensor::scalar(1.0)));
        let (x, y) = dataset.get_batch(&[2, 0]);
        assert_eq!(x, tensor![[5, 6], [1, 2]]);
        assert_eq!(y, tensor![0, 0]);
    }

    #[test]
    #[should_panic(expected = "must hold the same number of samples")]
    fn rejects_mismatched_tensors() {
        TensorDataset::new(Tensor::zeros([3, 2]), Tensor::zeros([2]));
    }
}
