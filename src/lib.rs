//! General purpose machine learning library.

pub mod activation;
pub mod linalg;
pub mod loss;
pub mod metrics;
pub mod nn;
pub mod optimizer;
pub mod regression;
pub mod statistics;
pub mod utils;

pub use activation::Activation;
pub use linalg::Tensor;
pub use loss::Loss;
pub use nn::{Dataset, Initializer};
pub use optimizer::Optimizer;

pub mod prelude;
