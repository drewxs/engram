use crate::{
    linalg::Tensor,
    nn::{Layer, Regularization},
    Activation, Initializer, Loss, Optimizer,
};

#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub d_weights: Option<Tensor>,
    pub d_biases: Option<Tensor>,
    pub input: Option<Tensor>,
    pub output: Option<Tensor>,
    pub activation: Activation,
    pub eval: bool,
}

impl DenseLayer {
    /// Create a new layer with the given number of input and output features.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::prelude::*;
    /// let layer = DenseLayer::new(3, 2, Initializer::Xavier, Activation::ReLU);
    /// # assert_eq!(layer.weights().shape(), (2, 3));
    /// # assert_eq!(layer.biases().shape(), (2, 1));
    /// ```
    pub fn new(
        f_in: usize,
        f_out: usize,
        initializer: Initializer,
        activation: Activation,
    ) -> Self {
        Self {
            weights: Tensor::initialize(f_out, f_in, &initializer),
            biases: Tensor::initialize(f_out, 1, &initializer),
            d_weights: None,
            d_biases: None,
            input: None,
            output: None,
            activation,
            eval: false,
        }
    }

    /// Create a new layer with the given number of input and output features,
    /// using Xavier initialization and ReLU activation as defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::prelude::*;
    /// let layer = DenseLayer::default(4, 7);
    /// # assert_eq!(layer.weights().shape(), (7, 4));
    /// # assert_eq!(layer.biases().shape(), (7, 1));
    /// ```
    pub fn default(f_in: usize, f_out: usize) -> Self {
        Self::new(f_in, f_out, Initializer::Xavier, Activation::ReLU)
    }
}

impl Layer for DenseLayer {
    fn weights(&self) -> &Tensor {
        &self.weights
    }

    fn biases(&self) -> &Tensor {
        &self.biases
    }

    /// Forward pass through the layer, returning the output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::prelude::*;
    /// let mut layer = DenseLayer::default(3, 2);
    /// let input = tensor![1.0, 2.0, 3.0];
    /// let output = layer.forward(&input);
    /// # assert_eq!(output.shape(), (1, 2));
    /// ```
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weights.transpose());
        let output = output
            .add(&self.biases)
            .resize(output.rows, output.cols) // resize implicit broadcast from `add`
            .activate(&self.activation);

        if !self.eval {
            self.input = Some(input.clone());
            self.output = Some(output.clone());
        }

        output
    }

    /// Backpropagate through the layer, returning the average loss.
    ///
    /// # Examples
    ///
    /// ```
    /// # use engram::prelude::*;
    /// let mut layer = DenseLayer::default(3, 2);
    /// let input = tensor![1.0, 2.0, 3.0];
    /// let output = layer.forward(&input);
    /// let target = tensor![1.0, 0.0];
    /// let loss = layer.backward(&target, &Loss::MSE);
    /// # assert!(loss > 0.0);
    /// ```
    fn backward(&mut self, target: &Tensor, loss_fn: &Loss) -> f64 {
        let predictions = self
            .output
            .as_ref()
            .expect("Call to back_propagate without calling feed_forward first!");
        let input = self
            .input
            .as_ref()
            .expect("Input not set before backward pass!");

        let loss = loss_fn.loss(predictions, target);
        let d_loss = loss_fn.grad(predictions, target);

        let d_activation = predictions.activate(&self.activation);
        let d_output = &d_activation * &d_loss;

        let d_weights = input.transpose().matmul(&d_output);
        let d_biases = d_output.sum_axis(0);

        if !self.eval {
            self.d_weights = Some(d_weights.clone());
            self.d_biases = Some(d_biases);
        }

        loss.mean()
    }

    fn update_parameters(&mut self, optimizer: &mut dyn Optimizer) {
        if let Some(d_weights) = &self.d_weights {
            optimizer.step(&mut self.weights, d_weights);
        }
        if let Some(d_biases) = &self.d_biases {
            optimizer.step(&mut self.biases, d_biases);
        }
    }

    fn regularization_loss(&self, reg: &dyn Regularization) -> f64 {
        reg.loss(&self.weights)
    }

    fn apply_regularization(&mut self, reg: &dyn Regularization) {
        let d_weights = reg.grad(&self.weights);
        let d_biases = reg.grad(&self.biases);

        for (w, reg_w) in self.weights.data.iter_mut().zip(d_weights.data.iter()) {
            for (w_elem, reg_w_elem) in w.iter_mut().zip(reg_w.iter()) {
                *w_elem -= reg_w_elem;
            }
        }

        for (b, reg_b) in self.biases.data.iter_mut().zip(d_biases.data.iter()) {
            for (b_elem, reg_b_elem) in b.iter_mut().zip(reg_b.iter()) {
                *b_elem -= reg_b_elem;
            }
        }
    }

    fn eval(&mut self) {
        self.eval = true;
    }

    fn train(&mut self) {
        self.eval = false;
    }
}
