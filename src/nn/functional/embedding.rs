use crate::Tensor;

/// Looks up rows of `weight` by id, like `torch.nn.functional.embedding`.
///
/// `input` holds integer ids in any shape, and the result has that shape followed by the embedding size. Gradients accumulate into the rows that were used, including repeats.
///
/// ```
/// # use engram::{nn::functional::embedding, tensor};
/// let weight = tensor![[0, 0], [1, 1], [2, 2]];
/// assert_eq!(embedding(&tensor![[2, 0]], &weight), tensor![[[2, 2], [0, 0]]]);
/// ```
///
/// # Panics
///
/// Panics if `weight` is not 2-d or an id is not a valid row index.
pub fn embedding(input: &Tensor, weight: &Tensor) -> Tensor {
    assert_eq!(
        weight.ndim(),
        2,
        "embedding: weight must be 2-d, got shape {:?}",
        weight.shape()
    );
    let rows = weight.shape()[0];
    let ids: Vec<usize> = input
        .data()
        .iter()
        .map(|&id| {
            assert!(
                id >= 0.0 && id.fract() == 0.0 && (id as usize) < rows,
                "embedding: id {id} is not an integer below {rows}"
            );
            id as usize
        })
        .collect();
    let mut shape = input.shape().to_vec();
    shape.push(weight.shape()[1]);
    weight.index_select(0, &ids).reshape(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    #[test]
    fn accumulates_gradients_for_repeated_ids() {
        let weight = Tensor::randn([3, 2]).set_requires_grad(true);
        embedding(&tensor![1, 1, 0], &weight).sum().backward();
        assert_eq!(weight.grad().unwrap(), tensor![[1, 1], [2, 2], [0, 0]]);
    }

    #[test]
    #[should_panic(expected = "embedding: id 3 is not an integer below 3")]
    fn rejects_out_of_range_ids() {
        embedding(&tensor![3], &Tensor::zeros([3, 2]));
    }
}
