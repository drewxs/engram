/// Returns the result of the leaky rectified linear unit function.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// assert_eq!(leaky_relu(0.0), 0.0, "leaky_relu(0.0)");
/// assert_eq!(leaky_relu(-10.0), -0.1, "leaky_relu(-10.0)");
/// ```
pub fn leaky_relu(x: f64) -> f64 {
    f64::max(x, 0.01 * x)
}

/// Returns the derivative of the leaky rectified linear unit function.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// assert_eq!(d_leaky_relu(0.0), 0.01, "d_leaky_relu(0.0)");
/// assert_eq!(d_leaky_relu(-10.0), 0.01, "d_leaky_relu(-10.0)");
/// ```
pub fn d_leaky_relu(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.01
    }
}
