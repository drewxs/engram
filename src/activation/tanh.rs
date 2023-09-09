/// Returns the result of the hyperbolic tangent function.
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Returns the derivative of the hyperbolic tangent function.
pub fn d_tanh(x: f64) -> f64 {
    1.0 - tanh(x).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh() {
        assert_eq!(tanh(0.0), 0.0, "tanh(0.0)");
        assert_eq!(tanh(1.0), 0.7615941559557649, "tanh(1.0)");
    }

    #[test]
    fn test_d_tanh() {
        assert_eq!(d_tanh(0.0), 1.0, "d_tanh(0.0)");
        assert_eq!(d_tanh(1.0), 0.41997434161402614, "d_tanh(1.0)");
    }
}
