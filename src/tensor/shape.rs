//! Shapes, dimension indices and broadcasting.

/// Types that describe the shape of a tensor, such as `[2, 3]` or `vec![4]`.
pub trait IntoShape {
    fn into_shape(self) -> Vec<usize>;
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self.clone()
    }
}

/// A dimension index. Signed indices count from the end, so `-1` is the last dimension.
pub trait Dim: Copy {
    /// Resolves the index for a tensor with `ndim` dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of range.
    fn resolve(self, ndim: usize) -> usize;
}

impl Dim for usize {
    fn resolve(self, ndim: usize) -> usize {
        assert!(
            self < ndim,
            "dimension {self} is out of range for a {ndim}-d tensor"
        );
        self
    }
}

macro_rules! impl_signed_dim {
    ($($t:ty),*) => {$(
        impl Dim for $t {
            fn resolve(self, ndim: usize) -> usize {
                let index = if self < 0 { self as isize + ndim as isize } else { self as isize };
                assert!(
                    (0..ndim as isize).contains(&index),
                    "dimension {self} is out of range for a {ndim}-d tensor"
                );
                index as usize
            }
        }
    )*};
}

impl_signed_dim!(i32, isize);

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major strides of a contiguous tensor.
pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (1..shape.len()).rev() {
        strides[i - 1] = strides[i] * shape[i];
    }
    strides
}

/// Splits a shape around `dim` into the number of leading elements, the size of `dim` and the number of trailing elements.
pub(crate) fn split_at_dim(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    (numel(&shape[..dim]), shape[dim], numel(&shape[dim + 1..]))
}

/// Returns the shape that `a` and `b` broadcast to, or `None` if they are incompatible.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let dim =
        |shape: &[usize], i: usize| (i + shape.len()).checked_sub(ndim).map_or(1, |j| shape[j]);
    (0..ndim)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y => Some(x),
            (1, y) => Some(y),
            (x, 1) => Some(x),
            _ => None,
        })
        .collect()
}

/// Strides for reading a tensor of `shape` as if it were broadcast to `target`. Broadcast dimensions get a stride of zero.
pub(crate) fn broadcast_strides(shape: &[usize], target: &[usize]) -> Vec<usize> {
    let offset = target.len() - shape.len();
    let own = strides(shape);
    (0..target.len())
        .map(|i| match i.checked_sub(offset) {
            Some(j) if shape[j] != 1 => own[j],
            _ => 0,
        })
        .collect()
}

/// Calls `f` once for every row of `shape`, meaning every run of elements along its last dimension, with the flat offset where that row starts in each tensor described by `strides`.
pub(crate) fn for_each_row<const N: usize>(
    shape: &[usize],
    strides: [&[usize]; N],
    mut f: impl FnMut([usize; N]),
) {
    if numel(shape) == 0 {
        return;
    }
    let lead = shape.len().saturating_sub(1);
    let mut index = vec![0; lead];
    let mut offsets = [0; N];
    loop {
        f(offsets);
        let mut d = lead;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            index[d] += 1;
            for (offset, strides) in offsets.iter_mut().zip(&strides) {
                *offset += strides[d];
            }
            if index[d] < shape[d] {
                break;
            }
            for (offset, strides) in offsets.iter_mut().zip(&strides) {
                *offset -= strides[d] * shape[d];
            }
            index[d] = 0;
        }
    }
}

/// Calls `f` for every element of `shape` in row-major order, with the flat offset of that element in each tensor described by `strides`.
pub(crate) fn for_each_offset<const N: usize>(
    shape: &[usize],
    strides: [&[usize]; N],
    mut f: impl FnMut([usize; N]),
) {
    let inner = shape.last().copied().unwrap_or(1);
    let steps: [usize; N] = std::array::from_fn(|i| strides[i].last().copied().unwrap_or(0));
    for_each_row(shape, strides, |starts| {
        for k in 0..inner {
            f(std::array::from_fn(|i| starts[i] + k * steps[i]));
        }
    });
}

/// Calls `f(start, stride)` for every lane along `dim`. A lane holds the elements whose indices differ only in `dim`, found at `start + k * stride` for `k` in `0..shape[dim]`.
pub(crate) fn for_each_lane(shape: &[usize], dim: usize, mut f: impl FnMut(usize, usize)) {
    let (outer, size, inner) = split_at_dim(shape, dim);
    for o in 0..outer {
        for i in 0..inner {
            f(o * size * inner + i, inner);
        }
    }
}

/// Calls `f` for every lane along `dim` with contiguous slices: the lane of `out` to write and the matching lanes of `inputs`. Lanes that are not contiguous in memory are gathered into scratch buffers and written back afterwards.
pub(crate) fn map_lanes<const N: usize>(
    shape: &[usize],
    dim: usize,
    out: &mut [f32],
    inputs: [&[f32]; N],
    mut f: impl FnMut(&mut [f32], [&[f32]; N]),
) {
    let (outer, size, inner) = split_at_dim(shape, dim);
    if inner == 1 {
        for o in 0..outer {
            let lane = o * size..(o + 1) * size;
            f(
                &mut out[lane.clone()],
                inputs.map(|input| &input[lane.clone()]),
            );
        }
        return;
    }
    let mut out_lane = vec![0.0; size];
    let mut input_lanes = vec![vec![0.0; size]; N];
    for_each_lane(shape, dim, |start, stride| {
        for (lane, input) in input_lanes.iter_mut().zip(&inputs) {
            for (k, value) in lane.iter_mut().enumerate() {
                *value = input[start + k * stride];
            }
        }
        f(
            &mut out_lane,
            std::array::from_fn(|i| input_lanes[i].as_slice()),
        );
        for (k, &value) in out_lane.iter().enumerate() {
            out[start + k * stride] = value;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_negative_dims() {
        assert_eq!((-1).resolve(3), 2);
        assert_eq!((-3).resolve(3), 0);
        assert_eq!(1usize.resolve(3), 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rejects_out_of_range_dims() {
        (-4).resolve(3);
    }

    #[test]
    fn computes_strides() {
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn broadcasts_shapes() {
        assert_eq!(broadcast_shape(&[4, 1, 3], &[2, 1]), Some(vec![4, 2, 3]));
        assert_eq!(broadcast_shape(&[], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 3], &[3, 3]), None);
    }

    #[test]
    fn computes_broadcast_strides() {
        assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
        assert_eq!(broadcast_strides(&[2, 1], &[2, 3]), vec![1, 0]);
    }
}
