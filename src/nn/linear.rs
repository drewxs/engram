use crate::{
    nn::{functional, Module},
    Tensor,
};

/// Applies `y = x Wᵀ + b` to the last dimension of the input.
///
/// Weights and biases start from `U(-1/√in, 1/√in)`, the same default as PyTorch.
///
/// ```
/// # use engram::{nn::{Linear, Module}, Tensor};
/// let layer = Linear::new(4, 2);
/// assert_eq!(layer.forward(&Tensor::randn([8, 4])).shape(), &[8, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct Linear {
    /// Shape `[out_features, in_features]`.
    pub weight: Tensor,
    /// Shape `[out_features]`.
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Linear {
        let bound = 1.0 / (in_features as f32).sqrt();
        Linear {
            bias: Some(Tensor::uniform([out_features], -bound, bound).set_requires_grad(true)),
            ..Linear::without_bias(in_features, out_features)
        }
    }

    /// Creates a layer with no bias term.
    pub fn without_bias(in_features: usize, out_features: usize) -> Linear {
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight = Tensor::uniform([out_features, in_features], -bound, bound);
        Linear {
            weight: weight.set_requires_grad(true),
            bias: None,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::linear(input, &self.weight, self.bias.as_ref())
    }

    fn parameters(&self) -> Vec<Tensor> {
        std::iter::once(self.weight.clone())
            .chain(self.bias.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_manual_affine_map() {
        let layer = Linear::new(3, 2);
        let x = Tensor::randn([5, 3]);
        let expected = x.matmul(&layer.weight.t()) + layer.bias.as_ref().unwrap();
        assert!(layer.forward(&x).allclose(&expected, 1e-5));
    }

    #[test]
    fn initializes_within_pytorch_bounds() {
        let layer = Linear::new(16, 8);
        let bound = 0.25;
        assert!(layer.weight.abs().max().item() <= bound);
        assert!(layer.bias.as_ref().unwrap().abs().max().item() <= bound);
        assert_eq!(layer.parameters().len(), 2);
        assert_eq!(Linear::without_bias(16, 8).parameters().len(), 1);
    }
}
