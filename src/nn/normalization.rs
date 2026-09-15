use crate::{
    nn::{functional, Module},
    tensor::IntoShape,
    Tensor,
};

/// Layer normalization over the trailing dimensions given by `normalized_shape`, with a learned elementwise scale and offset.
///
/// ```
/// # use engram::{nn::{LayerNorm, Module}, Tensor};
/// let norm = LayerNorm::new([8]);
/// let y = norm.forward(&(Tensor::randn([4, 10, 8]) * 3.0));
/// assert!(y.mean_dim(-1, false).abs().max().item() < 1e-5);
/// ```
#[derive(Clone, Debug)]
pub struct LayerNorm {
    /// Scale applied after normalizing, starting at ones.
    pub weight: Option<Tensor>,
    /// Offset applied after scaling, starting at zeros.
    pub bias: Option<Tensor>,
    normalized_shape: Vec<usize>,
    eps: f32,
}

impl LayerNorm {
    /// Creates a layer with a learned scale starting at ones and an offset starting at zeros.
    pub fn new(normalized_shape: impl IntoShape) -> LayerNorm {
        let shape = normalized_shape.into_shape();
        LayerNorm {
            weight: Some(Tensor::ones(&shape).set_requires_grad(true)),
            bias: Some(Tensor::zeros(&shape).set_requires_grad(true)),
            ..LayerNorm::without_affine(shape)
        }
    }

    /// Creates a layer that normalizes without a learned scale and offset.
    pub fn without_affine(normalized_shape: impl IntoShape) -> LayerNorm {
        LayerNorm {
            weight: None,
            bias: None,
            normalized_shape: normalized_shape.into_shape(),
            eps: 1e-5,
        }
    }

    /// Sets the value added to the variance for numerical stability.
    pub fn eps(mut self, eps: f32) -> LayerNorm {
        self.eps = eps;
        self
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::layer_norm(
            input,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        [("weight", &self.weight), ("bias", &self.bias)]
            .into_iter()
            .filter_map(|(name, parameter)| Some((name.to_string(), parameter.clone()?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_affine_parameters() {
        let norm = LayerNorm::new([3, 4]);
        let names: Vec<String> = norm
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, ["weight", "bias"]);
        assert_eq!(norm.num_parameters(), 24);
        assert!(LayerNorm::without_affine([4]).parameters().is_empty());
    }

    #[test]
    fn trains_scale_and_offset() {
        let norm = LayerNorm::new([2]);
        norm.forward(&crate::tensor![[1, 3], [2, 6]])
            .sum()
            .backward();
        assert!(norm.weight.as_ref().unwrap().grad().is_some());
        assert_eq!(
            norm.bias.as_ref().unwrap().grad().unwrap(),
            crate::tensor![2, 2]
        );
    }
}
