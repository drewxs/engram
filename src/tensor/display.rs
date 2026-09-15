use std::fmt;

use crate::Tensor;

/// Tensors with more elements than this only print the first and last few entries of each dimension.
const SUMMARIZE_THRESHOLD: usize = 1000;
const EDGE_ITEMS: usize = 3;
const PREFIX: &str = "tensor(";

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.data();
        let width = data
            .iter()
            .map(|x| format!("{x:.4}").len())
            .max()
            .unwrap_or(0);
        write!(f, "{PREFIX}")?;
        if self.ndim() == 0 {
            write!(f, "{:.4}", data[0])?;
        } else {
            let layout = Layout {
                width,
                summarize: data.len() > SUMMARIZE_THRESHOLD,
            };
            layout.write(f, &data, self.shape(), 0)?;
        }
        if self.requires_grad() {
            write!(f, ", requires_grad=true")?;
        }
        write!(f, ")")
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

struct Layout {
    width: usize,
    summarize: bool,
}

impl Layout {
    fn write(
        &self,
        f: &mut fmt::Formatter<'_>,
        data: &[f32],
        shape: &[usize],
        depth: usize,
    ) -> fmt::Result {
        let len = shape[0];
        let stride: usize = shape[1..].iter().product();
        write!(f, "[")?;
        for (k, index) in self.visible(len).into_iter().enumerate() {
            if k > 0 {
                if shape.len() == 1 {
                    write!(f, ", ")?;
                } else {
                    write!(f, ",{}", "\n".repeat(shape.len() - 1))?;
                    write!(f, "{:indent$}", "", indent = PREFIX.len() + depth + 1)?;
                }
            }
            match index {
                None => write!(f, "...")?,
                Some(i) if shape.len() == 1 => {
                    write!(f, "{:>width$.4}", data[i], width = self.width)?
                }
                Some(i) => self.write(
                    f,
                    &data[i * stride..(i + 1) * stride],
                    &shape[1..],
                    depth + 1,
                )?,
            }
        }
        write!(f, "]")
    }

    /// Indices to print along a dimension, with `None` marking the elided middle.
    fn visible(&self, len: usize) -> Vec<Option<usize>> {
        if self.summarize && len > 2 * EDGE_ITEMS {
            (0..EDGE_ITEMS)
                .map(Some)
                .chain([None])
                .chain((len - EDGE_ITEMS..len).map(Some))
                .collect()
        } else {
            (0..len).map(Some).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor, Tensor};

    #[test]
    fn formats_like_pytorch() {
        assert_eq!(tensor![1, 2].to_string(), "tensor([1.0000, 2.0000])");
        assert_eq!(
            tensor![[1, -2], [3, 4]].to_string(),
            "tensor([[ 1.0000, -2.0000],\n        [ 3.0000,  4.0000]])"
        );
        assert_eq!(Tensor::scalar(0.5).to_string(), "tensor(0.5000)");
        assert_eq!(Tensor::zeros([1, 1, 1]).to_string(), "tensor([[[0.0000]]])");
    }

    #[test]
    fn marks_tensors_that_require_grad() {
        let t = tensor![1].set_requires_grad(true);
        assert_eq!(t.to_string(), "tensor([1.0000], requires_grad=true)");
    }

    #[test]
    fn summarizes_large_tensors() {
        let s = Tensor::zeros([2000]).to_string();
        assert_eq!(
            s,
            "tensor([0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000])"
        );
    }
}
