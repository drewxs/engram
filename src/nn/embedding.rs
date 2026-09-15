use crate::{
    nn::{functional, Module},
    Tensor,
};

/// A table of learned vectors indexed by integer ids, like `torch.nn.Embedding`. Vectors start from a standard normal distribution.
///
/// ```
/// # use engram::{nn::{Embedding, Module}, tensor};
/// let embedding = Embedding::new(100, 16);
/// assert_eq!(embedding.forward(&tensor![[4, 8, 15], [16, 23, 42]]).shape(), &[2, 3, 16]);
/// ```
#[derive(Clone, Debug)]
pub struct Embedding {
    /// Shape `[num_embeddings, embedding_dim]`.
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Embedding {
        Embedding {
            weight: Tensor::randn([num_embeddings, embedding_dim]).set_requires_grad(true),
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::embedding(input, &self.weight)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}
