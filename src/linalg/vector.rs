/// Dot product of two vectors.
///
/// # Examples
///
/// ```
/// # use engram::linalg::dot;
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.0, 2.0, 3.0];
/// assert_eq!(dot(&a, &b), 14.0);
/// ```
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0, |acc, (x, y)| acc + x * y)
}

/// Magnitude of a vector.
///
/// # Examples
///
/// ```
/// # use engram::linalg::magnitude;
/// let a = [1.0, 2.0, 3.0];
/// assert!((magnitude(&a) - 3.7417).abs() < 1e-4);
/// ```
pub fn magnitude(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |acc, x| acc + x * x).sqrt()
}
