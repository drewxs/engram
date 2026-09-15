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
        affine_parameters(self.weight.as_ref(), self.bias.as_ref())
    }
}

/// Batch normalization over the channel dimension of inputs shaped `[N, C, ...]`, like `torch.nn.BatchNorm1d` and `torch.nn.BatchNorm2d`.
///
/// While training, each channel is normalized with the statistics of the current batch and the running estimates are updated. Evaluation mode uses the running estimates instead, so call [`Module::eval`] before inference.
///
/// ```
/// # use engram::{nn::{BatchNorm2d, Module}, Tensor};
/// let mut norm = BatchNorm2d::new(3);
/// assert_eq!(norm.forward(&Tensor::randn([8, 3, 4, 4])).shape(), &[8, 3, 4, 4]);
/// norm.eval();
/// assert_eq!(norm.state_dict().len(), 4);
/// ```
#[derive(Clone, Debug)]
pub struct BatchNorm {
    /// Scale applied after normalizing, starting at ones.
    pub weight: Option<Tensor>,
    /// Offset applied after scaling, starting at zeros.
    pub bias: Option<Tensor>,
    /// Running estimate of each channel's mean, starting at zeros.
    pub running_mean: Tensor,
    /// Running estimate of each channel's variance, starting at ones.
    pub running_var: Tensor,
    momentum: f32,
    eps: f32,
    training: bool,
}

/// Batch normalization for inputs shaped `[N, C]` or `[N, C, L]`.
pub type BatchNorm1d = BatchNorm;

/// Batch normalization for inputs shaped `[N, C, H, W]`.
pub type BatchNorm2d = BatchNorm;

impl BatchNorm {
    /// Creates a layer for `num_features` channels with a momentum of 0.1.
    pub fn new(num_features: usize) -> BatchNorm {
        BatchNorm {
            weight: Some(Tensor::ones([num_features]).set_requires_grad(true)),
            bias: Some(Tensor::zeros([num_features]).set_requires_grad(true)),
            running_mean: Tensor::zeros([num_features]),
            running_var: Tensor::ones([num_features]),
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }

    /// Sets how far the running statistics move toward the statistics of each batch.
    pub fn momentum(mut self, momentum: f32) -> BatchNorm {
        self.momentum = momentum;
        self
    }

    /// Sets the value added to the variance for numerical stability.
    pub fn eps(mut self, eps: f32) -> BatchNorm {
        self.eps = eps;
        self
    }
}

impl Module for BatchNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::batch_norm(
            input,
            Some((&self.running_mean, &self.running_var)),
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.training,
            self.momentum,
            self.eps,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        affine_parameters(self.weight.as_ref(), self.bias.as_ref())
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        vec![
            ("running_mean".to_string(), self.running_mean.clone()),
            ("running_var".to_string(), self.running_var.clone()),
        ]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

fn affine_parameters(weight: Option<&Tensor>, bias: Option<&Tensor>) -> Vec<(String, Tensor)> {
    [("weight", weight), ("bias", bias)]
        .into_iter()
        .filter_map(|(name, parameter)| Some((name.to_string(), parameter?.clone())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{Linear, Sequential},
        tensor,
    };

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
        norm.forward(&tensor![[1, 3], [2, 6]]).sum().backward();
        assert!(norm.weight.as_ref().unwrap().grad().is_some());
        assert_eq!(norm.bias.as_ref().unwrap().grad().unwrap(), tensor![2, 2]);
    }

    #[test]
    fn batch_norm_switches_statistics_with_training_mode() {
        let mut norm = BatchNorm2d::new(2).momentum(0.5);
        let x = Tensor::randn([8, 2, 3, 3]) + 5.0;
        let y = norm.forward(&x);
        assert!(y.mean().abs().item() < 1e-4);
        assert!(norm.running_mean.min().item() > 2.0);

        norm.eval();
        let z = norm.forward(&x);
        let mean = norm.running_mean.reshape([1, 2, 1, 1]);
        let std = (&norm.running_var + 1e-5).sqrt().reshape([1, 2, 1, 1]);
        assert!(z.allclose(&((&x - &mean) / &std), 1e-4));
    }

    #[test]
    fn batch_norm_state_includes_running_statistics() {
        let build = || {
            Sequential::new()
                .add(Linear::new(3, 4))
                .add(BatchNorm1d::new(4))
        };
        let model = build();
        model.forward(&Tensor::randn([6, 3]));
        let state = model.state_dict();
        assert!(state.contains_key("1.running_mean") && state.contains_key("1.running_var"));
        let restored = build();
        restored.load_state_dict(&state).unwrap();
        assert_eq!(restored.state_dict(), state);
    }
}
