/// Compute the L1 norm (Manhattan distance) of a vector.
///
/// # Examples
///
/// ```
/// # use engram::metrics::norm_l1;
/// let x = [1.0, 2.0, 3.0];
/// assert_eq!(norm_l1(&x), 6.0);
/// ```
pub fn norm_l1(vector: &[f64]) -> f64 {
    vector.iter().fold(0.0, |acc, x| acc + x.abs())
}

/// Compute the L2 norm (Euclidean distance) of a vector.
///
/// # Examples
///
/// ```
/// # use engram::metrics::norm_l2;
/// let x = [1.0, 2.0, 3.0];
/// assert_eq!(norm_l2(&x), 3.7416573867739413);
/// ```
pub fn norm_l2(vector: &[f64]) -> f64 {
    vector.iter().fold(0.0, |acc, x| acc + x * x).sqrt()
}

/// Compute the L∞ (max) norm of a vector.
///
/// # Examples
///
/// ```
/// # use engram::metrics::norm_linf;
/// let x = [1.0, 2.0, 3.0];
/// assert_eq!(norm_linf(&x), 3.0);
/// ```
pub fn norm_linf(vector: &[f64]) -> f64 {
    vector.iter().fold(0.0, |acc, x| f64::max(acc, x.abs()))
}
