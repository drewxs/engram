/// Returns the result of the sigmoid function.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Returns the derivative of the sigmoid function.
pub fn d_sigmoid(x: f64) -> f64 {
    let y = sigmoid(x);
    y * (1.0 - y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5, "sigmoid(0.0)");
        assert_eq!(sigmoid(1.0), 0.7310585786300049, "sigmoid(1.0)");
    }

    #[test]
    fn test_d_sigmoid() {
        assert_eq!(d_sigmoid(0.0), 0.25, "d_sigmoid(0.0)");
        assert_eq!(d_sigmoid(1.0), 0.19661193324148185, "d_sigmoid(1.0)");
    }
}
