//! N-dimensional tensors.

mod activation;
mod binary;
mod create;
mod display;
mod manip;
mod matmul;
mod reduce;
pub(crate) mod shape;
pub(crate) mod storage;
mod unary;

pub(crate) use activation::sigmoid;
pub(crate) use matmul::batched_gemm;
pub use shape::{Dim, IntoShape};
pub use storage::empty_cache;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, PoisonError, RwLock,
};

use crate::autograd::{self, BackwardFn, Node};
use storage::CollectBuffer;

/// An n-dimensional array of `f32` values that can record gradients.
///
/// Tensors are cheap to clone, and a clone shares the same data, so an update made through one handle is visible through all of them. Operations return new tensors. When any input requires a gradient, the result also remembers how it was computed, and calling [`Tensor::backward`] on a scalar result fills in [`Tensor::grad`] for every leaf that contributed to it.
///
/// ```
/// # use engram::{tensor, Tensor};
/// let t = tensor![[1, 2, 3], [4, 5, 6]];
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
#[derive(Clone)]
pub struct Tensor(Arc<Inner>);

struct Inner {
    shape: Vec<usize>,
    data: RwLock<Arc<Vec<f32>>>,
    grad: RwLock<Option<Tensor>>,
    requires_grad: AtomicBool,
    node: Option<Node>,
}

impl Tensor {
    /// Creates a tensor from flat data in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` does not match the number of elements in `shape`.
    pub fn from_vec(data: Vec<f32>, shape: impl IntoShape) -> Tensor {
        let shape = shape.into_shape();
        assert_eq!(
            data.len(),
            shape::numel(&shape),
            "cannot create a tensor of shape {shape:?} from {} values",
            data.len()
        );
        Tensor::from_parts(Arc::new(data), shape)
    }

    /// Creates a zero-dimensional tensor holding a single value.
    pub fn scalar(value: f32) -> Tensor {
        Tensor::from_vec(vec![value], Vec::new())
    }

    pub(crate) fn from_parts(data: Arc<Vec<f32>>, shape: Vec<usize>) -> Tensor {
        Tensor(Arc::new(Inner {
            shape,
            data: RwLock::new(data),
            grad: RwLock::new(None),
            requires_grad: AtomicBool::new(false),
            node: None,
        }))
    }

    /// Records `self` as the output of an operation on `inputs` when any of them requires a gradient.
    ///
    /// `backward` receives the gradient of the output and returns one gradient per input, in the same order. It may skip inputs that do not require a gradient by returning `None`. Capture a clone of `self` rather than the returned tensor when the backward pass needs the output, since `self` is not part of the graph and cannot form a reference cycle.
    pub(crate) fn record<F>(self, inputs: &[&Tensor], backward: F) -> Tensor
    where
        F: Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync + 'static,
    {
        if !autograd::is_grad_enabled() || !inputs.iter().any(|t| t.requires_grad()) {
            return self;
        }
        let backward: BackwardFn = Box::new(backward);
        Tensor(Arc::new(Inner {
            shape: self.0.shape.clone(),
            data: RwLock::new(self.data()),
            grad: RwLock::new(None),
            requires_grad: AtomicBool::new(true),
            node: Some(Node {
                inputs: inputs.iter().map(|&t| t.clone()).collect(),
                backward,
            }),
        }))
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn ndim(&self) -> usize {
        self.0.shape.len()
    }

    pub fn numel(&self) -> usize {
        shape::numel(&self.0.shape)
    }

    /// Returns the size of one dimension.
    pub fn size(&self, dim: impl Dim) -> usize {
        self.0.shape[dim.resolve(self.ndim())]
    }

    /// Returns a snapshot of the underlying buffer.
    pub(crate) fn data(&self) -> Arc<Vec<f32>> {
        self.0
            .data
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Mutates the buffer in place, copying it first if another tensor still holds it.
    pub(crate) fn update(&self, f: impl FnOnce(&mut [f32])) {
        let mut data = self.0.data.write().unwrap_or_else(PoisonError::into_inner);
        let buffer = Arc::make_mut(&mut *data);
        f(buffer);
    }

    /// Copies the values of `src` into this tensor without recording gradients.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    pub fn copy_from(&self, src: &Tensor) {
        assert_eq!(
            self.shape(),
            src.shape(),
            "copy_from expects matching shapes"
        );
        let src = src.data();
        self.update(|dst| dst.copy_from_slice(&src));
    }

    /// Returns the values in row-major order.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data().to_vec()
    }

    /// Returns the value of a single-element tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor does not have exactly one element.
    pub fn item(&self) -> f32 {
        assert_eq!(
            self.numel(),
            1,
            "item() expects a single-element tensor, got shape {:?}",
            self.shape()
        );
        self.data()[0]
    }

    /// Returns true if every element is within `tolerance` of the matching element of `other`.
    pub fn allclose(&self, other: &Tensor, tolerance: f32) -> bool {
        self.shape() == other.shape()
            && self
                .data()
                .iter()
                .zip(other.data().iter())
                .all(|(a, b)| (a - b).abs() <= tolerance)
    }

    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.load(Ordering::Relaxed)
    }

    /// Sets whether operations on this tensor record gradients and returns the tensor for chaining.
    ///
    /// # Panics
    ///
    /// Panics if the tensor was computed by a tracked operation. Call [`Tensor::detach`] first.
    pub fn set_requires_grad(&self, requires_grad: bool) -> Tensor {
        assert!(
            self.is_leaf(),
            "set_requires_grad can only be called on leaf tensors"
        );
        self.0.requires_grad.store(requires_grad, Ordering::Relaxed);
        self.clone()
    }

    /// Returns true if the tensor was created directly rather than computed by a tracked operation.
    pub fn is_leaf(&self) -> bool {
        self.0.node.is_none()
    }

    /// Returns the gradient accumulated by [`Tensor::backward`].
    pub fn grad(&self) -> Option<Tensor> {
        self.0
            .grad
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Clears the accumulated gradient.
    pub fn zero_grad(&self) {
        *self.0.grad.write().unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// Returns a tensor that shares this tensor's data but is excluded from gradient tracking.
    pub fn detach(&self) -> Tensor {
        Tensor::from_parts(self.data(), self.0.shape.clone())
    }

    pub(crate) fn node(&self) -> Option<&Node> {
        self.0.node.as_ref()
    }

    pub(crate) fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub(crate) fn accumulate_grad(&self, grad: Tensor) {
        let mut slot = self.0.grad.write().unwrap_or_else(PoisonError::into_inner);
        *slot = Some(match slot.take() {
            Some(existing) => add_same_shape(&existing, &grad),
            // One gradient tensor can reach several inputs, so each leaf keeps its own handle.
            None => grad.detach(),
        });
    }
}

/// Adds two tensors of the same shape without recording gradients.
pub(crate) fn add_same_shape(a: &Tensor, b: &Tensor) -> Tensor {
    let (x, y) = (a.data(), b.data());
    let data = x.iter().zip(y.iter()).map(|(x, y)| x + y).collect_buffer();
    Tensor::from_vec(data, a.shape())
}

impl Drop for Inner {
    // Dropping a long graph would otherwise recurse once per operation and can overflow the stack, so nodes that are no longer shared are unlinked iteratively.
    fn drop(&mut self) {
        let data = self.data.get_mut().unwrap_or_else(PoisonError::into_inner);
        if let Some(buffer) = Arc::get_mut(data) {
            storage::recycle(std::mem::take(buffer));
        }
        let Some(node) = self.node.take() else {
            return;
        };
        drop(node.backward);
        let mut pending = node.inputs;
        while let Some(tensor) = pending.pop() {
            if let Some(mut inner) = Arc::into_inner(tensor.0) {
                if let Some(node) = inner.node.take() {
                    drop(node.backward);
                    pending.extend(node.inputs);
                }
            }
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        self.shape() == other.shape() && self.data() == other.data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    #[test]
    fn clones_share_data() {
        let a = Tensor::zeros([2, 2]);
        let b = a.clone();
        a.update(|data| data[0] = 5.0);
        assert_eq!(b.to_vec(), vec![5.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn detached_tensors_copy_on_write() {
        let a = Tensor::zeros([2]);
        let b = a.detach();
        a.update(|data| data[0] = 1.0);
        assert_eq!(a.to_vec(), vec![1.0, 0.0]);
        assert_eq!(b.to_vec(), vec![0.0, 0.0]);
    }

    #[test]
    fn copies_values_in_place() {
        let a = Tensor::zeros([2, 2]);
        a.copy_from(&tensor![[1, 2], [3, 4]]);
        assert_eq!(a, tensor![[1, 2], [3, 4]]);
    }

    #[test]
    fn reads_items_and_sizes() {
        let t = Tensor::zeros([2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.size(-1), 4);
        assert_eq!(Tensor::scalar(3.5).item(), 3.5);
    }

    #[test]
    #[should_panic(expected = "cannot create a tensor of shape [2, 2] from 3 values")]
    fn rejects_mismatched_data() {
        Tensor::from_vec(vec![1.0, 2.0, 3.0], [2, 2]);
    }

    #[test]
    fn tensors_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tensor>();
    }

    #[test]
    fn releases_the_graph_once_outputs_drop() {
        let w = Tensor::randn([3, 3]).set_requires_grad(true);
        let loss = (w.matmul(&w).relu() * 2.0).sum();
        assert!(Arc::strong_count(&w.0) > 1);
        loss.backward();
        drop(loss);
        assert_eq!(Arc::strong_count(&w.0), 1);
        assert_eq!(Arc::strong_count(&w.data()), 2);
    }

    #[test]
    fn handles_empty_tensors() {
        let empty = Tensor::zeros([0, 3]);
        assert_eq!((&empty + &Tensor::ones([3])).shape(), &[0, 3]);
        assert_eq!(empty.sum().item(), 0.0);
        assert_eq!(empty.sum_dim(0, false), Tensor::zeros([3]));
        assert_eq!(empty.matmul(&Tensor::ones([3, 2])).shape(), &[0, 2]);
        assert_eq!(empty.softmax(-1).shape(), &[0, 3]);
        assert_eq!(
            Tensor::cat(&[empty.clone(), Tensor::ones([2, 3])], 0),
            Tensor::ones([2, 3])
        );
        assert_eq!(empty.to_string(), "tensor([])");
    }
}
