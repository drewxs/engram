/// Returns the result of the rectified linear unit function.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// assert_eq!(relu(0.0), 0.0, "relu(0.0)");
/// assert_eq!(relu(10.0), 10.0, "relu(10.0)");
/// ```
pub fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

/// Returns the derivative of the rectified linear unit function.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// assert_eq!(d_relu(0.0), 0.0, "d_relu(0.0)");
/// assert_eq!(d_relu(-10.0), 0.0, "d_relu(-10.0)");
/// ```
pub fn d_relu(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
