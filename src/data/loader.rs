use crate::{
    data::{Collate, Dataset},
    random,
};

/// Iterates over a dataset in batches, like `torch.utils.data.DataLoader`.
///
/// Every call to [`DataLoader::iter`] starts a new epoch and reshuffles when shuffling is enabled.
///
/// ```
/// # use engram::{data::{DataLoader, TensorDataset}, Tensor};
/// let dataset = TensorDataset::new(Tensor::randn([10, 3]), Tensor::zeros([10]));
/// let loader = DataLoader::new(dataset, 4).shuffle(true);
/// let sizes: Vec<usize> = loader.iter().map(|(x, _)| x.shape()[0]).collect();
/// assert_eq!(sizes, vec![4, 4, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct DataLoader<D> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<D: Dataset> DataLoader<D>
where
    D::Item: Collate,
{
    /// # Panics
    ///
    /// Panics if `batch_size` is zero.
    pub fn new(dataset: D, batch_size: usize) -> Self {
        assert!(batch_size > 0, "DataLoader batch size must be positive");
        DataLoader {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    /// Visits samples in a new random order every epoch.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Skips the final batch when it is smaller than the batch size.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Returns the dataset being loaded.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Returns the number of batches in an epoch.
    pub fn len(&self) -> usize {
        let samples = self.dataset.len();
        if self.drop_last {
            samples / self.batch_size
        } else {
            samples.div_ceil(self.batch_size)
        }
    }

    /// Returns true if an epoch has no batches.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Starts a new epoch.
    pub fn iter(&self) -> Batches<'_, D> {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            random::shuffle(&mut indices);
        }
        Batches {
            loader: self,
            indices,
            position: 0,
        }
    }
}

impl<'a, D: Dataset> IntoIterator for &'a DataLoader<D>
where
    D::Item: Collate,
{
    type Item = D::Item;
    type IntoIter = Batches<'a, D>;

    fn into_iter(self) -> Batches<'a, D> {
        self.iter()
    }
}

/// The batches of one epoch, created by [`DataLoader::iter`].
pub struct Batches<'a, D> {
    loader: &'a DataLoader<D>,
    indices: Vec<usize>,
    position: usize,
}

impl<D: Dataset> Iterator for Batches<'_, D>
where
    D::Item: Collate,
{
    type Item = D::Item;

    fn next(&mut self) -> Option<D::Item> {
        let size = (self.indices.len() - self.position).min(self.loader.batch_size);
        if size == 0 || (self.loader.drop_last && size < self.loader.batch_size) {
            return None;
        }
        let batch = &self.indices[self.position..self.position + size];
        self.position += size;
        Some(self.loader.dataset.get_batch(batch))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        data::{DataLoader, TensorDataset},
        manual_seed, Tensor,
    };

    fn dataset(n: usize) -> TensorDataset {
        let x = Tensor::arange(0.0, n as f32, 1.0);
        TensorDataset::new(x.clone(), x)
    }

    #[test]
    fn batches_in_order_by_default() {
        let loader = DataLoader::new(dataset(5), 2);
        let batches: Vec<Vec<f32>> = loader.iter().map(|(x, _)| x.to_vec()).collect();
        assert_eq!(batches, vec![vec![0.0, 1.0], vec![2.0, 3.0], vec![4.0]]);
        assert_eq!(loader.len(), 3);
    }

    #[test]
    fn drops_incomplete_final_batch() {
        let loader = DataLoader::new(dataset(5), 2).drop_last(true);
        assert_eq!(loader.iter().count(), 2);
        assert_eq!(loader.len(), 2);
    }

    #[test]
    fn shuffles_every_epoch_and_keeps_pairs_aligned() {
        manual_seed(5);
        let loader = DataLoader::new(dataset(50), 8).shuffle(true);
        let first: Vec<f32> = loader
            .iter()
            .flat_map(|(x, y)| {
                assert_eq!(x, y);
                x.to_vec()
            })
            .collect();
        let second: Vec<f32> = (&loader)
            .into_iter()
            .flat_map(|(x, _)| x.to_vec())
            .collect();
        assert_ne!(first, second);
        let mut sorted = first.clone();
        sorted.sort_by(f32::total_cmp);
        assert_eq!(sorted, Tensor::arange(0.0, 50.0, 1.0).to_vec());
    }
}
