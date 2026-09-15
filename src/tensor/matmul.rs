//! Matrix multiplication.

use crate::{
    tensor::{binary::sum_to, shape},
    Tensor,
};

impl Tensor {
    /// Matrix product with NumPy semantics.
    ///
    /// Tensors with more than two dimensions are stacks of matrices in their last two dimensions, and the leading dimensions broadcast. A 1-d argument acts as a row vector on the left or a column vector on the right, and that dimension is dropped from the result.
    ///
    /// ```
    /// # use engram::{tensor, Tensor};
    /// let a = tensor![[1, 2], [3, 4]];
    /// assert_eq!(a.matmul(&tensor![[5], [6]]), tensor![[17], [39]]);
    /// assert_eq!(Tensor::zeros([8, 2, 3]).matmul(&Tensor::zeros([3, 4])).shape(), &[8, 2, 4]);
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(
            self.ndim() > 0 && other.ndim() > 0,
            "matmul: arguments must have at least one dimension"
        );
        match (self.ndim(), other.ndim()) {
            (1, 1) => self
                .unsqueeze(0)
                .matmul(&other.unsqueeze(1))
                .reshape(Vec::new()),
            (1, _) => self.unsqueeze(0).matmul(other).squeeze(-2),
            (_, 1) => self.matmul(&other.unsqueeze(1)).squeeze(-1),
            _ => {
                let out = batched_gemm(self, false, other, false);
                let (a, b) = (self.clone(), other.clone());
                out.record(&[self, other], move |g| {
                    vec![
                        a.requires_grad()
                            .then(|| sum_to(&batched_gemm(g, false, &b, true), a.shape())),
                        b.requires_grad()
                            .then(|| sum_to(&batched_gemm(&a, true, g, false), b.shape())),
                    ]
                })
            }
        }
    }
}

/// Computes `op(a) @ op(b)` over broadcast batch dimensions without recording gradients, where `op` swaps the last two dimensions when its flag is set. Transposes are expressed through strides, so no data is copied.
pub(crate) fn batched_gemm(a: &Tensor, transpose_a: bool, b: &Tensor, transpose_b: bool) -> Tensor {
    let (a_batch, [a_rows, a_cols]) = split_matrix(a.shape());
    let (b_batch, [b_rows, b_cols]) = split_matrix(b.shape());
    let (n, m) = if transpose_a {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };
    let (m_b, p) = if transpose_b {
        (b_cols, b_rows)
    } else {
        (b_rows, b_cols)
    };
    assert_eq!(
        m,
        m_b,
        "matmul: shapes {:?} and {:?} are not aligned",
        a.shape(),
        b.shape()
    );
    let batch = shape::broadcast_shape(a_batch, b_batch).unwrap_or_else(|| {
        panic!(
            "matmul: batch dimensions of {:?} and {:?} cannot be broadcast",
            a.shape(),
            b.shape()
        )
    });

    let mut out_shape = batch.clone();
    out_shape.extend([n, p]);
    let mut out = vec![0.0; shape::numel(&out_shape)];
    if out.is_empty() {
        return Tensor::from_vec(out, out_shape);
    }

    let a_strides = if transpose_a {
        (1, a_cols)
    } else {
        (a_cols, 1)
    };
    let b_strides = if transpose_b {
        (1, b_cols)
    } else {
        (b_cols, 1)
    };
    let (x, y) = (a.data(), b.data());
    let (a_size, b_size) = (a_rows * a_cols, b_rows * b_cols);
    let mut chunks = out.chunks_exact_mut(n * p);
    shape::for_each_offset(
        &batch,
        [
            &shape::broadcast_strides(a_batch, &batch),
            &shape::broadcast_strides(b_batch, &batch),
        ],
        |[i, j]| {
            let a = &x[i * a_size..(i + 1) * a_size];
            let b = &y[j * b_size..(j + 1) * b_size];
            gemm(
                [n, m, p],
                a,
                a_strides,
                b,
                b_strides,
                chunks.next().unwrap(),
            );
        },
    );
    Tensor::from_vec(out, out_shape)
}

fn split_matrix(shape: &[usize]) -> (&[usize], [usize; 2]) {
    let (batch, matrix) = shape.split_at(shape.len() - 2);
    (batch, [matrix[0], matrix[1]])
}

/// Writes the `n` by `p` product of an `n` by `m` matrix and an `m` by `p` matrix into `c`. Each input is read with its `(row, column)` strides.
fn gemm(
    [n, m, p]: [usize; 3],
    a: &[f32],
    (rsa, csa): (usize, usize),
    b: &[f32],
    (rsb, csb): (usize, usize),
    c: &mut [f32],
) {
    if m == 0 {
        c.fill(0.0);
        return;
    }
    assert!(
        a.len() > (n - 1) * rsa + (m - 1) * csa
            && b.len() > (m - 1) * rsb + (p - 1) * csb
            && c.len() == n * p,
        "gemm: buffers are too small for the requested product"
    );
    // SAFETY: the assertion above guarantees every element addressed through the strides lies inside its buffer, and `c` does not alias the inputs because it is borrowed mutably.
    unsafe {
        matrixmultiply::sgemm(
            n,
            m,
            p,
            1.0,
            a.as_ptr(),
            rsa as isize,
            csa as isize,
            b.as_ptr(),
            rsb as isize,
            csb as isize,
            0.0,
            c.as_mut_ptr(),
            p as isize,
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{autograd::testing::assert_gradients, tensor, Tensor};

    fn naive_matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let (n, m, p) = (a.shape()[0], a.shape()[1], b.shape()[1]);
        let (x, y) = (a.to_vec(), b.to_vec());
        let mut out = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                out[i * p + j] = (0..m).map(|k| x[i * m + k] * y[k * p + j]).sum();
            }
        }
        Tensor::from_vec(out, [n, p])
    }

    #[test]
    fn multiplies_matrices() {
        let a = tensor![[1, 2, 3], [4, 5, 6]];
        let b = tensor![[7, 8], [9, 10], [11, 12]];
        assert_eq!(a.matmul(&b), tensor![[58, 64], [139, 154]]);
        let (x, y) = (Tensor::randn([17, 9]), Tensor::randn([9, 13]));
        assert!(x.matmul(&y).allclose(&naive_matmul(&x, &y), 1e-4));
    }

    #[test]
    fn handles_vectors() {
        let m = tensor![[1, 2], [3, 4]];
        let v = tensor![1, 1];
        assert_eq!(v.matmul(&v), Tensor::scalar(2.0));
        assert_eq!(m.matmul(&v), tensor![3, 7]);
        assert_eq!(v.matmul(&m), tensor![4, 6]);
    }

    #[test]
    fn broadcasts_batch_dimensions() {
        let a = Tensor::randn([2, 1, 3, 4]);
        let b = Tensor::randn([5, 4, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 5, 3, 2]);
        let expected = naive_matmul(&a.select(0, 1).select(0, 0), &b.select(0, 3));
        assert!(c.select(0, 1).select(0, 3).allclose(&expected, 1e-4));
    }

    #[test]
    #[should_panic(expected = "not aligned")]
    fn rejects_misaligned_shapes() {
        Tensor::zeros([2, 3]).matmul(&Tensor::zeros([2, 3]));
    }

    #[test]
    fn gradients_of_matmul() {
        assert_gradients([&Tensor::randn([3, 4]), &Tensor::randn([4, 2])], |t| {
            t[0].matmul(&t[1])
        });
        assert_gradients(
            [&Tensor::randn([2, 1, 3, 4]), &Tensor::randn([5, 4, 2])],
            |t| t[0].matmul(&t[1]),
        );
        assert_gradients([&Tensor::randn([3]), &Tensor::randn([3, 2])], |t| {
            t[0].matmul(&t[1])
        });
    }
}
