use crate::{
    nn::{
        functional::{self, Size2},
        Module,
    },
    Tensor,
};

/// 2-d convolution over images of shape `[N, C, H, W]`, like `torch.nn.Conv2d`.
///
/// Weights and biases start from `U(-1/√fan_in, 1/√fan_in)` with `fan_in = in_channels * kh * kw`, the same default as PyTorch.
///
/// ```
/// # use engram::{nn::{Conv2d, Module}, Tensor};
/// let conv = Conv2d::new(3, 16, 3).padding(1);
/// assert_eq!(conv.forward(&Tensor::randn([2, 3, 28, 28])).shape(), &[2, 16, 28, 28]);
/// ```
#[derive(Clone, Debug)]
pub struct Conv2d {
    /// Shape `[out_channels, in_channels, kh, kw]`.
    pub weight: Tensor,
    /// Shape `[out_channels]`.
    pub bias: Option<Tensor>,
    stride: Size2,
    padding: Size2,
    dilation: Size2,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: impl Into<Size2>) -> Conv2d {
        let kernel = kernel_size.into();
        let fan_in = in_channels * kernel.height * kernel.width;
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight_shape = [out_channels, in_channels, kernel.height, kernel.width];
        Conv2d {
            weight: Tensor::uniform(weight_shape, -bound, bound).set_requires_grad(true),
            bias: Some(Tensor::uniform([out_channels], -bound, bound).set_requires_grad(true)),
            stride: Size2::from(1),
            padding: Size2::from(0),
            dilation: Size2::from(1),
        }
    }

    pub fn stride(mut self, stride: impl Into<Size2>) -> Conv2d {
        self.stride = stride.into();
        self
    }

    /// Sets the number of zeros added on each side of the input.
    pub fn padding(mut self, padding: impl Into<Size2>) -> Conv2d {
        self.padding = padding.into();
        self
    }

    /// Sets the spacing between kernel elements.
    pub fn dilation(mut self, dilation: impl Into<Size2>) -> Conv2d {
        self.dilation = dilation.into();
        self
    }

    /// Removes the bias term.
    pub fn without_bias(mut self) -> Conv2d {
        self.bias = None;
        self
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::conv2d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
        )
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
    fn applies_options() {
        let conv = Conv2d::new(2, 4, (3, 5))
            .stride(2)
            .padding((1, 2))
            .without_bias();
        assert_eq!(
            conv.forward(&Tensor::randn([1, 2, 9, 9])).shape(),
            &[1, 4, 5, 5]
        );
        assert_eq!(conv.parameters().len(), 1);
        let bound = 1.0 / 30f32.sqrt();
        assert!(conv.weight.abs().max().item() <= bound);
    }
}
