use crate::{Tensor, Tensor2D};

impl From<Tensor2D> for Tensor {
    fn from(data: Tensor2D) -> Self {
        Tensor {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }
}
