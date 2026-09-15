use crate::{
    nn::{
        functional::{self, Size2},
        Module,
    },
    Tensor,
};

/// 2-d max pooling over images of shape `[N, C, H, W]`, like `torch.nn.MaxPool2d`. The stride defaults to the kernel size.
///
/// ```
/// # use engram::{nn::{MaxPool2d, Module}, Tensor};
/// let pool = MaxPool2d::new(2);
/// assert_eq!(pool.forward(&Tensor::randn([4, 8, 32, 32])).shape(), &[4, 8, 16, 16]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MaxPool2d {
    kernel_size: Size2,
    stride: Size2,
    padding: Size2,
}

impl MaxPool2d {
    pub fn new(kernel_size: impl Into<Size2>) -> MaxPool2d {
        let kernel_size = kernel_size.into();
        MaxPool2d {
            kernel_size,
            stride: kernel_size,
            padding: Size2::from(0),
        }
    }

    pub fn stride(mut self, stride: impl Into<Size2>) -> MaxPool2d {
        self.stride = stride.into();
        self
    }

    /// Sets the padding added on each side, which may be at most half the kernel size.
    pub fn padding(mut self, padding: impl Into<Size2>) -> MaxPool2d {
        self.padding = padding.into();
        self
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::max_pool2d(input, self.kernel_size, self.stride, self.padding)
    }
}

/// 2-d average pooling over images of shape `[N, C, H, W]`, like `torch.nn.AvgPool2d`. The stride defaults to the kernel size.
#[derive(Clone, Copy, Debug)]
pub struct AvgPool2d {
    kernel_size: Size2,
    stride: Size2,
    padding: Size2,
}

impl AvgPool2d {
    pub fn new(kernel_size: impl Into<Size2>) -> AvgPool2d {
        let kernel_size = kernel_size.into();
        AvgPool2d {
            kernel_size,
            stride: kernel_size,
            padding: Size2::from(0),
        }
    }

    pub fn stride(mut self, stride: impl Into<Size2>) -> AvgPool2d {
        self.stride = stride.into();
        self
    }

    /// Sets the padding added on each side, which may be at most half the kernel size.
    pub fn padding(mut self, padding: impl Into<Size2>) -> AvgPool2d {
        self.padding = padding.into();
        self
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::avg_pool2d(input, self.kernel_size, self.stride, self.padding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_match_functional_pooling() {
        let x = Tensor::randn([2, 3, 6, 6]);
        assert_eq!(
            MaxPool2d::new(3).forward(&x),
            functional::max_pool2d(&x, 3, 3, 0)
        );
        let avg = AvgPool2d::new(2).stride(1).padding(1);
        assert_eq!(avg.forward(&x), functional::avg_pool2d(&x, 2, 1, 1));
    }
}
