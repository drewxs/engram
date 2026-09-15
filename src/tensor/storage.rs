//! Reuse of tensor buffers.
//!
//! Training runs the same shapes every step, so the large buffers freed by one step are exactly the ones the next step asks for. Keeping them in a per-thread cache avoids the page faults the system allocator pays when it hands memory back to the OS and grows again.

use std::{cell::RefCell, collections::HashMap};

/// Buffers smaller than this are left to the system allocator, which already recycles small allocations cheaply.
const MIN_CACHED_LEN: usize = 1 << 14;

/// Upper bound on the memory cached by one thread.
const MAX_CACHED_BYTES: usize = 1 << 30;

thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
}

#[derive(Default)]
struct Cache {
    buffers: HashMap<usize, Vec<Vec<f32>>>,
    bytes: usize,
}

/// Returns an empty buffer with capacity for `len` values, reusing a cached one when available.
pub(crate) fn with_capacity(len: usize) -> Vec<f32> {
    if len >= MIN_CACHED_LEN {
        let cached = CACHE.try_with(|cache| {
            let mut cache = cache.borrow_mut();
            let buffer = cache.buffers.get_mut(&len)?.pop()?;
            cache.bytes -= bytes(len);
            Some(buffer)
        });
        if let Ok(Some(mut buffer)) = cached {
            buffer.clear();
            return buffer;
        }
    }
    Vec::with_capacity(len)
}

/// Returns a buffer holding `len` copies of `value`.
pub(crate) fn full(len: usize, value: f32) -> Vec<f32> {
    let mut buffer = with_capacity(len);
    buffer.resize(len, value);
    buffer
}

pub(crate) fn zeros(len: usize) -> Vec<f32> {
    full(len, 0.0)
}

/// Keeps a buffer that no tensor references anymore for later reuse.
pub(crate) fn recycle(buffer: Vec<f32>) {
    let len = buffer.capacity();
    if len < MIN_CACHED_LEN {
        return;
    }
    // The cache is already gone while the thread shuts down, and the buffer is simply freed then.
    let _ = CACHE.try_with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.bytes + bytes(len) <= MAX_CACHED_BYTES {
            cache.bytes += bytes(len);
            cache.buffers.entry(len).or_default().push(buffer);
        }
    });
}

/// Frees every tensor buffer cached by the current thread.
///
/// Large buffers released by dropped tensors are kept for reuse by later operations of the same size. Call this after training to hand that memory back to the system.
pub fn empty_cache() {
    let _ = CACHE.try_with(|cache| *cache.borrow_mut() = Cache::default());
}

fn bytes(len: usize) -> usize {
    len * std::mem::size_of::<f32>()
}

/// Collects values into a buffer drawn from the cache.
pub(crate) trait CollectBuffer: Iterator<Item = f32> + Sized {
    fn collect_buffer(self) -> Vec<f32> {
        let (lower, upper) = self.size_hint();
        let mut buffer = with_capacity(upper.unwrap_or(lower));
        buffer.extend(self);
        buffer
    }
}

impl<I: Iterator<Item = f32>> CollectBuffer for I {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn reuses_buffers_of_dropped_tensors() {
        let len = MIN_CACHED_LEN * 2;
        let t = Tensor::zeros([len]);
        let address = t.data().as_ptr();
        drop(t);
        let u = Tensor::ones([len]);
        assert_eq!(u.data().as_ptr(), address);
        assert!(u.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn keeps_shared_buffers_alive() {
        let len = MIN_CACHED_LEN * 2;
        let t = Tensor::zeros([len]);
        let view = t.reshape([2, MIN_CACHED_LEN]);
        drop(t);
        let u = Tensor::ones([len]);
        assert_ne!(u.data().as_ptr(), view.data().as_ptr());
        assert!(view.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn caches_only_large_buffers_until_emptied() {
        recycle(Vec::with_capacity(8));
        recycle(Vec::with_capacity(MIN_CACHED_LEN));
        CACHE.with(|cache| assert_eq!(cache.borrow().bytes, bytes(MIN_CACHED_LEN)));
        empty_cache();
        CACHE.with(|cache| assert_eq!(cache.borrow().bytes, 0));
    }
}
