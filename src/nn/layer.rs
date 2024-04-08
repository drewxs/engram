use crate::{linalg::Tensor, Activation, Initializer, Loss};

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub d_weights: Option<Tensor>,
    pub d_biases: Option<Tensor>,
    pub input: Option<Tensor>,
    pub output: Option<Tensor>,
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with the given number of input and output features.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let layer = Layer::new(3, 2, Initializer::Xavier, Activation::ReLU);
    /// assert_eq!(layer.weights.shape(), (2, 3));
    /// assert_eq!(layer.biases.shape(), (2, 1));
    /// ```
    pub fn new(
        f_in: usize,
        f_out: usize,
        initializer: Initializer,
        activation: Activation,
    ) -> Layer {
        Layer {
            weights: Tensor::initialize(f_out, f_in, &initializer),
            biases: Tensor::initialize(f_out, 1, &initializer),
            d_weights: None,
            d_biases: None,
            input: None,
            output: None,
            activation,
        }
    }

    /// Create a new layer with the given number of input and output features,
    /// using Xavier initialization and ReLU activation as defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let layer = Layer::default(4, 7);
    /// assert_eq!(layer.weights.shape(), (7, 4));
    /// assert_eq!(layer.biases.shape(), (7, 1));
    /// ```
    pub fn default(f_in: usize, f_out: usize) -> Layer {
        Layer::new(f_in, f_out, Initializer::Xavier, Activation::ReLU)
    }

    /// Forward pass through the layer, returning the output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut layer = Layer::default(3, 2);
    /// let input = tensor![1.0, 2.0, 3.0];
    /// let output = layer.forward(&input);
    /// assert_eq!(output.shape(), (1, 2));
    /// ```
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weights.transpose());
        let output = output
            .add(&self.biases)
            .resize(output.rows, output.cols)
            .activate(&self.activation);

        self.input = Some(input.clone());
        self.output = Some(output.clone());

        output
    }

    /// Backpropagate through the layer, returning the average loss.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::*;
    /// let mut layer = Layer::default(3, 2);
    /// let input = tensor![1.0, 2.0, 3.0];
    /// let output = layer.forward(&input);
    /// let target = tensor![1.0, 0.0];
    /// let loss = layer.backward(&target, &Loss::MSE);
    /// assert!(loss > 0.0);
    /// ```
    pub fn backward(&mut self, target: &Tensor, loss_fn: &Loss) -> f64 {
        let predictions = self
            .output
            .as_ref()
            .expect("Call to back_propagate without calling feed_forward first!");
        let input = self
            .input
            .as_ref()
            .expect("Input not set before backward pass!");

        let loss = loss_fn.loss(&predictions, &target);
        let d_loss = loss_fn.grad(&predictions, &target);

        let d_activation = predictions.activate(&self.activation);
        let d_output = &d_activation * &d_loss;

        let d_weights = input.transpose().matmul(&d_output);
        self.d_weights = Some(d_weights);

        let d_biases = d_output.sum_axis(0);
        self.d_biases = Some(d_biases);

        loss.mean()
    }
}
