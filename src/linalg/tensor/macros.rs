/// Creates a new `Tensor` from a two-dimensional vector of floating point values.
///
/// # Usage
///
/// ```
/// # use engram::*;
/// let tensor = tensor![[1.0, 2.0], [3.0, 4.0]];
/// assert_eq!(tensor.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// ```
#[macro_export]
macro_rules! tensor {
    ( $( [ $( $x:expr ),* ] ),* ) => {{
        let mut data = vec![];
        $(
            data.push(vec![$($x as f64),*]);
        )*
        $crate::Tensor::from(data)
    }};

    ( $( $x:expr ),* $(,)? ) => {{
        let data = vec![vec![$($x as f64),*]];
        $crate::Tensor::from(data)
    }};
}
