//! Choosing between elements with masks and comparisons.

use crate::{
    tensor::{
        binary::{sum_to, zip_broadcast},
        shape,
        storage::CollectBuffer,
    },
    Tensor,
};

impl Tensor {
    /// Picks elements from `on_true` where this mask is nonzero and from `on_false` elsewhere, broadcasting all three tensors, like `torch.where`.
    ///
    /// Gradients flow back to `on_true` and `on_false` through the positions picked from each.
    ///
    /// ```
    /// # use engram::{tensor, Tensor};
    /// let x = tensor![-1, 2, -3];
    /// assert_eq!(x.greater(0.0).where_cond(&x, &Tensor::scalar(0.0)), tensor![0, 2, 0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the shapes cannot be broadcast together.
    pub fn where_cond(&self, on_true: &Tensor, on_false: &Tensor) -> Tensor {
        let shape = [on_true.shape(), on_false.shape()]
            .into_iter()
            .try_fold(self.shape().to_vec(), |acc, s| {
                shape::broadcast_shape(&acc, s)
            })
            .unwrap_or_else(|| {
                panic!(
                    "where_cond: shapes {:?}, {:?} and {:?} cannot be broadcast together",
                    self.shape(),
                    on_true.shape(),
                    on_false.shape()
                )
            });
        let expand = |t: &Tensor| t.detach().broadcast_to(&shape).data();
        let (mask, a, b) = (expand(self), expand(on_true), expand(on_false));
        let data = mask
            .iter()
            .zip(a.iter().zip(b.iter()))
            .map(|(&m, (&a, &b))| if m != 0.0 { a } else { b })
            .collect_buffer();

        let (mask, x, y) = (self.detach(), on_true.clone(), on_false.clone());
        Tensor::from_vec(data, shape).record(&[on_true, on_false], move |g| {
            let pick = |keep: bool| {
                zip_broadcast(g, &mask, "where_cond", move |g, m| {
                    if (m != 0.0) == keep {
                        g
                    } else {
                        0.0
                    }
                })
            };
            vec![
                x.requires_grad().then(|| sum_to(&pick(true), x.shape())),
                y.requires_grad().then(|| sum_to(&pick(false), y.shape())),
            ]
        })
    }

    /// Replaces elements with `value` where `mask` is nonzero, like `Tensor.masked_fill`.
    ///
    /// ```
    /// # use engram::tensor;
    /// let scores = tensor![[1, 2], [3, 4]];
    /// let mask = tensor![[0, 1], [0, 0]];
    /// assert_eq!(scores.masked_fill(&mask, -1.0), tensor![[1, -1], [3, 4]]);
    /// ```
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor {
        mask.where_cond(&Tensor::scalar(value), self)
    }

    /// Elementwise maximum with broadcasting, like `torch.maximum`. Where both inputs are equal the gradient is split evenly between them.
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        self.extremum(other, "maximum", |a, b| a > b)
    }

    /// Elementwise minimum with broadcasting, like `torch.minimum`. Where both inputs are equal the gradient is split evenly between them.
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        self.extremum(other, "minimum", |a, b| a < b)
    }

    fn extremum(&self, other: &Tensor, op: &'static str, better: fn(f32, f32) -> bool) -> Tensor {
        let out = zip_broadcast(self, other, op, |a, b| if better(b, a) { b } else { a });
        let (a, b) = (self.clone(), other.clone());
        out.record(&[self, other], move |g| {
            let share = zip_broadcast(&a, &b, op, |x, y| {
                if better(x, y) {
                    1.0
                } else if better(y, x) {
                    0.0
                } else {
                    0.5
                }
            });
            vec![
                a.requires_grad().then(|| sum_to(&(g * &share), a.shape())),
                b.requires_grad()
                    .then(|| sum_to(&(g * (1.0 - &share)), b.shape())),
            ]
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{autograd::testing::assert_gradients, tensor, Tensor};

    #[test]
    fn selects_with_broadcasting() {
        let mask = tensor![[1], [0]];
        let picked = mask.where_cond(&tensor![1, 2, 3], &Tensor::scalar(-1.0));
        assert_eq!(picked, tensor![[1, 2, 3], [-1, -1, -1]]);

        let scores = tensor![[0.5, 1.5], [2.5, 3.5]];
        let masked = scores.masked_fill(&tensor![0, 1], f32::NEG_INFINITY);
        assert_eq!(masked.softmax(-1).to_vec(), vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn takes_elementwise_extremes() {
        let (a, b) = (tensor![1, 5, 3], tensor![4, 2, 3]);
        assert_eq!(a.maximum(&b), tensor![4, 5, 3]);
        assert_eq!(a.minimum(&tensor![2]), tensor![1, 2, 2]);
    }

    #[test]
    fn gradients_of_selection() {
        let mask = tensor![[1, 0, 1], [0, 1, 1]];
        let (x, y) = (Tensor::randn([2, 3]), Tensor::randn([3]));
        assert_gradients([&x, &y], |t| mask.where_cond(&t[0], &t[1]));
        assert_gradients([&x], |t| t[0].masked_fill(&mask, 2.0));

        // The inputs stay at least 1 apart, so finite differences never cross a tie.
        let (a, b) = (tensor![-2, 0.5, 3], tensor![1, -1, 2]);
        assert_gradients([&a, &b], |t| t[0].maximum(&t[1]));
        assert_gradients([&a, &b], |t| t[0].minimum(&t[1]));
    }

    #[test]
    fn splits_gradients_evenly_on_ties() {
        let a = tensor![1].set_requires_grad(true);
        let b = tensor![1].set_requires_grad(true);
        a.maximum(&b).sum().backward();
        assert_eq!(
            (a.grad().unwrap().item(), b.grad().unwrap().item()),
            (0.5, 0.5)
        );
    }
}
