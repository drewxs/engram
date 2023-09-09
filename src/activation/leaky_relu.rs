/// Returns the result of the leaky rectified linear unit function.
pub fn leaky_relu(x: f64) -> f64 {
    f64::max(x, 0.01 * x)
}

/// Returns the derivative of the leaky rectified linear unit function.
pub fn d_leaky_relu(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(0.0), 0.0, "leaky_relu(0.0)");
        assert_eq!(leaky_relu(-10.0), -0.1, "leaky_relu(-10.0)");
    }

    #[test]
    fn test_d_leaky_relu() {
        assert_eq!(d_leaky_relu(0.0), 0.01, "d_leaky_relu(0.0)");
        assert_eq!(d_leaky_relu(-10.0), 0.01, "d_leaky_relu(-10.0)");
    }
}
