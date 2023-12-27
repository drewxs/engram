use crate::Tensor;

/// Cholesky decomposition of a symmetric, positive-definite matrix.
/// Returns the product of the lower triangular matrix and its conjugate transpose.
/// Returns None if the input matrix is not not square or positive-definite.
///
/// # Examples
/// ```
/// # use engram::{linalg, tensor};
/// let t1 = tensor![[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]];
/// let t2 = linalg::cholesky(&t1).unwrap();
/// assert_eq!(t2, tensor![[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]);
/// ```
pub fn cholesky(tensor: &Tensor) -> Option<Tensor> {
    if !tensor.is_square() {
        return None;
    }

    let mut result = Tensor::zeros(tensor.rows, tensor.cols);

    for i in 0..tensor.rows {
        for j in 0..(i + 1) {
            let mut sum = 0.0;

            if j == i {
                for k in 0..j {
                    sum += result.data[j][k] * result.data[j][k];
                }
                result.data[j][j] = (tensor.data[j][j] - sum).sqrt();
            } else {
                for k in 0..j {
                    sum += result.data[i][k] * result.data[j][k];
                }
                if result.data[j][j] > f64::EPSILON {
                    result.data[i][j] = (tensor.data[i][j] - sum) / result.data[j][j];
                } else {
                    return None;
                }
            }
        }
    }

    Some(result)
}

/// LU decomposition of a square matrix.
/// Returns the product of the lower triangular matrix and an upper triangular matrix.
///
/// # Examples
/// ```
/// # use engram::{linalg, tensor};
/// let t = tensor![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
/// let (l, u) = linalg::lu(&t).unwrap();
/// assert_eq!(l, tensor![[1.0, 0.0, 0.0], [-0.5, 1.0, 0.0], [0.0, -0.6666666666666666, 1.0]]);
/// assert_eq!(u, tensor![[2.0, -1.0, 0.0], [0.0, 1.5, -1.0], [0.0, 0.0, 1.3333333333333335]]);
/// ```
pub fn lu(tensor: &Tensor) -> Option<(Tensor, Tensor)> {
    if !tensor.is_square() {
        return None;
    }

    let n = tensor.data.len();

    let mut l = Tensor::identity(n);
    let mut u = tensor.clone();

    for i in 0..tensor.rows - 1 {
        for j in i + 1..tensor.rows {
            let factor = u.data[j][i] / u.data[i][i];
            l.data[j][i] = factor;

            for k in i..tensor.cols {
                u.data[j][k] -= factor * u.data[i][k];
            }
        }
    }

    Some((l, u))
}
