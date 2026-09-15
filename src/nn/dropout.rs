use crate::{
    nn::{functional, Module},
    Tensor,
};

/// Randomly zeroes elements while training. See [`functional::dropout`].
#[derive(Clone, Debug)]
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    /// Creates a dropout layer that zeroes each element with probability `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside `[0, 1]`.
    pub fn new(p: f32) -> Dropout {
        assert!(
            (0.0..=1.0).contains(&p),
            "dropout probability must be in [0, 1], got {p}"
        );
        Dropout { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::dropout(input, self.p, self.training)
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}
