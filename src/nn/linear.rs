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
    /// Creates a layer mapping `in_features` inputs to `out_features` outputs.
    pub fn new(in_features: usize, out_features: usize) -> Linear {
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight = Tensor::uniform([out_features, in_features], -bound, bound);
        let bias = Tensor::uniform([out_features], -bound, bound);
        Linear {
            weight: weight.set_requires_grad(true),
            bias: Some(bias.set_requires_grad(true)),
        }
    }

    /// Removes the bias term.
    pub fn without_bias(mut self) -> Linear {
        self.bias = None;
        self
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::linear(input, &self.weight, self.bias.as_ref())
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut parameters = vec![("weight".to_string(), self.weight.clone())];
        parameters.extend(self.bias.clone().map(|bias| ("bias".to_string(), bias)));
        parameters
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
        assert_eq!(Linear::new(16, 8).without_bias().parameters().len(), 1);
    }
}
