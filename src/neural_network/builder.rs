//! Sequential neural network builder.
//!
//! Allows creating a network using a builder pattern.

use crate::{Activation, Initializer, Layer, Loss, Network, Optimizer};

/// A builder for creating a `Network`.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let network = Sequential::new(&[2, 3, 1])
///    .activation(Activation::ReLU)
///    .loss_function(Loss::MeanSquaredError)
///    .optimizer(Optimizer::SGD { learning_rate: 0.1 })
///    .build();
/// assert_eq!(network.layers[0].weights.shape(), (2, 3));
/// assert_eq!(network.layers[1].weights.shape(), (3, 1));
/// assert_eq!(network.layers[0].activation, Activation::ReLU);
/// assert_eq!(network.loss_function, Loss::MeanSquaredError);
/// assert_eq!(network.optimizer, Optimizer::SGD { learning_rate: 0.1 })
/// ```
pub struct NetworkBuilder {
    layer_sizes: Vec<usize>,
    initializer: Option<Initializer>,
    activation: Option<Activation>,
    loss_function: Option<Loss>,
    optimizer: Option<Optimizer>,
}

impl NetworkBuilder {
    /// Creates a new `Sequential` with the specified layer sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Sequential::new(&[2, 3, 1]).build();
    /// assert_eq!(network.layers[0].weights.shape(), (2, 3));
    /// assert_eq!(network.layers[1].weights.shape(), (3, 1));
    /// ```
    pub fn new(layer_sizes: &[usize]) -> Self {
        NetworkBuilder {
            layer_sizes: layer_sizes.to_vec(),
            initializer: None,
            activation: None,
            loss_function: None,
            optimizer: None,
        }
    }

    /// Sets the initializer for the network.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Sequential::new(&[2, 3, 1]).initializer(Initializer::Xavier).build();
    /// assert_eq!(network.initializer, Initializer::Xavier);
    /// ```
    pub fn initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = Some(initializer);
        self
    }

    /// Sets the activation function for the network.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Sequential::new(&[2, 3, 1]).activation(Activation::ReLU).build();
    /// assert_eq!(network.layers[0].activation, Activation::ReLU);
    /// ```
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Sets the loss function for the network.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Sequential::new(&[2, 3, 1]).loss_function(Loss::MeanSquaredError).build();
    /// assert_eq!(network.loss_function, Loss::MeanSquaredError);
    /// ```
    pub fn loss_function(mut self, loss_function: Loss) -> Self {
        self.loss_function = Some(loss_function);
        self
    }

    /// Sets the optimizer for the network.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Sequential::new(&[2, 3, 1]).optimizer(Optimizer::SGD { learning_rate: 0.01 }).build();
    /// assert_eq!(network.optimizer, Optimizer::SGD { learning_rate: 0.01 });
    /// ```
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Builds the network.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let network = Sequential::new(&[2, 3, 1]).build();
    /// assert_eq!(network.layers[0].weights.shape(), (2, 3));
    /// assert_eq!(network.layers[1].weights.shape(), (3, 1));
    /// assert_eq!(network.layers[0].activation, Activation::Sigmoid);
    /// assert_eq!(network.loss_function, Loss::MeanSquaredError);
    /// assert_eq!(network.optimizer, Optimizer::SGD { learning_rate: 0.1 })
    /// ```
    pub fn build(self) -> Network {
        let initializer = self.initializer.unwrap_or(Initializer::Xavier);
        let activation = self.activation.unwrap_or(Activation::Sigmoid);
        let loss_function = self.loss_function.unwrap_or(Loss::MeanSquaredError);
        let optimizer = self
            .optimizer
            .unwrap_or(Optimizer::SGD { learning_rate: 0.1 });

        let mut layers = Vec::new();

        for i in 0..self.layer_sizes.len() - 1 {
            let layer = Layer::new(
                self.layer_sizes[i],
                self.layer_sizes[i + 1],
                &initializer,
                activation.clone(),
            );
            layers.push(layer);
        }

        Network::new(
            &self.layer_sizes,
            initializer,
            activation,
            loss_function,
            optimizer,
        )
    }
}
