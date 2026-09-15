use crate::{
    nn::{prefixed, Module},
    Tensor,
};

/// Chains modules so that each one's output feeds the next.
///
/// ```
/// # use engram::{nn::{self, Module}, Tensor};
/// let model = nn::Sequential::new()
///     .add(nn::Linear::new(4, 8))
///     .add(nn::Relu)
///     .add(nn::Linear::new(8, 1));
/// assert_eq!(model.forward(&Tensor::randn([2, 4])).shape(), &[2, 1]);
/// assert_eq!(model.parameters().len(), 4);
/// ```
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Creates an empty model.
    pub fn new() -> Sequential {
        Sequential::default()
    }

    /// Appends a module and returns the model, for building it in one expression.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, module: impl Module + 'static) -> Sequential {
        self.push(module);
        self
    }

    /// Appends a module.
    pub fn push(&mut self, module: impl Module + 'static) {
        self.layers.push(Box::new(module));
    }

    /// Returns the number of modules.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if the model holds no modules.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers
            .iter()
            .fold(input.clone(), |x, layer| layer.forward(&x))
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| prefixed(&i.to_string(), layer.named_parameters()))
            .collect()
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| prefixed(&i.to_string(), layer.named_buffers()))
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Dropout, Linear, Relu};

    #[test]
    fn chains_layers_and_closures() {
        let model = Sequential::new()
            .add(Linear::new(3, 4))
            .add(Relu)
            .add(|x: &Tensor| x * 2.0)
            .add(Tensor::sigmoid);
        let y = model.forward(&Tensor::randn([6, 3]));
        assert_eq!(y.shape(), &[6, 4]);
        assert!(y.min().item() >= 0.5);
        assert_eq!(model.len(), 4);
        assert_eq!(model.num_parameters(), 3 * 4 + 4);
    }

    #[test]
    fn propagates_training_mode() {
        let mut model = Sequential::new().add(Dropout::new(0.9));
        let x = Tensor::ones([1000]);
        assert_ne!(model.forward(&x), x);
        model.eval();
        assert_eq!(model.forward(&x), x);
        model.train();
        assert_ne!(model.forward(&x), x);
    }

    #[test]
    fn zero_grad_clears_parameters() {
        let model = Sequential::new().add(Linear::new(2, 1));
        model.forward(&Tensor::ones([1, 2])).sum().backward();
        assert!(model.parameters().iter().all(|p| p.grad().is_some()));
        model.zero_grad();
        assert!(model.parameters().iter().all(|p| p.grad().is_none()));
    }
}
