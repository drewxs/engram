/// Returns the result of the hyperbolic tangent function.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// assert_eq!(tanh(0.0), 0.0, "tanh(0.0)");
/// assert_eq!(tanh(1.0), 0.7615941559557649, "tanh(1.0)");
/// ```
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Returns the derivative of the hyperbolic tangent function.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// assert_eq!(d_tanh(0.0), 1.0, "d_tanh(0.0)");
/// assert_eq!(d_tanh(1.0), 0.41997434161402614, "d_tanh(1.0)");
/// ```
pub fn d_tanh(x: f64) -> f64 {
    1.0 - tanh(x).powi(2)
}
