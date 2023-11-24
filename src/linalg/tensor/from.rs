use crate::Tensor;

impl From<Vec<Vec<f64>>> for Tensor {
    fn from(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        Tensor { rows, cols, data }
    }
}
