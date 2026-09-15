//! Matrix decompositions. These are computed in double precision and are not differentiable.

use crate::Tensor;

/// Cholesky decomposition of a symmetric positive definite matrix.
///
/// Returns the lower triangular `L` with `A = L Lᵀ`, or `None` when the matrix is not square or not positive definite.
///
/// ```
/// # use engram::{linalg, tensor};
/// let a = tensor![[4, 12, -16], [12, 37, -43], [-16, -43, 98]];
/// let l = linalg::cholesky(&a).unwrap();
/// assert_eq!(l, tensor![[2, 0, 0], [6, 1, 0], [-8, 5, 3]]);
/// ```
pub fn cholesky(a: &Tensor) -> Option<Tensor> {
    let n = square_size(a)?;
    let a = a.data();
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            let value = a[i * n + j] as f64 - sum;
            if i == j {
                if value <= 0.0 {
                    return None;
                }
                l[i * n + i] = value.sqrt();
            } else {
                l[i * n + j] = value / l[j * n + j];
            }
        }
    }
    Some(to_tensor(l, n))
}

/// LU decomposition without pivoting.
///
/// Returns a unit lower triangular `L` and an upper triangular `U` with `A = L U`, or `None` when the matrix is not square or a zero pivot appears.
///
/// ```
/// # use engram::{linalg, tensor};
/// let a = tensor![[2, -1, 0], [-1, 2, -1], [0, -1, 2]];
/// let (l, u) = linalg::lu(&a).unwrap();
/// assert!(l.matmul(&u).allclose(&a, 1e-6));
/// assert!(u.allclose(&tensor![[2, -1, 0], [0, 1.5, -1], [0, 0, 4.0 / 3.0]], 1e-6));
/// ```
pub fn lu(a: &Tensor) -> Option<(Tensor, Tensor)> {
    let n = square_size(a)?;
    let mut u: Vec<f64> = a.data().iter().map(|&x| x as f64).collect();
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        l[i * n + i] = 1.0;
        let pivot = u[i * n + i];
        if pivot == 0.0 && i + 1 < n {
            return None;
        }
        for j in i + 1..n {
            let factor = u[j * n + i] / pivot;
            l[j * n + i] = factor;
            for k in i..n {
                u[j * n + k] -= factor * u[i * n + k];
            }
        }
    }
    Some((to_tensor(l, n), to_tensor(u, n)))
}

fn square_size(a: &Tensor) -> Option<usize> {
    (a.ndim() == 2 && a.shape()[0] == a.shape()[1]).then(|| a.shape()[0])
}

fn to_tensor(data: Vec<f64>, n: usize) -> Tensor {
    Tensor::from_vec(data.into_iter().map(|x| x as f32).collect(), [n, n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    #[test]
    fn cholesky_reconstructs_the_matrix() {
        let b = Tensor::randn([5, 5]);
        let a = b.matmul(&b.t()) + Tensor::eye(5);
        let l = cholesky(&a).unwrap();
        assert!(l.matmul(&l.t()).allclose(&a, 1e-4));
    }

    #[test]
    fn rejects_unsupported_matrices() {
        assert!(cholesky(&tensor![[1, 2], [2, 1]]).is_none());
        assert!(cholesky(&Tensor::zeros([2, 3])).is_none());
        assert!(lu(&tensor![[0, 1], [1, 0]]).is_none());
    }
}
