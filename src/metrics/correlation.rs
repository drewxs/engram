/// Calculate the Pearson correlation coefficient between two vectors.
/// Formula:
/// ```text
///     n * sum(x * y) - sum(x) * sum(y)
///     --------------------------------
///     sqrt((n * sum(x^2) - sum(x)^2) * (n * sum(y^2) - sum(y)^2))
/// ```
///
/// # Examples
///
/// ```
/// use engram::metrics::pearson_correlation;
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = [1.0, 2.1, 2.9, 3.9, 5.1];
/// assert_eq!(pearson_correlation(&a, &b), 0.998005980069749);
/// ```
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_x_sq: f64 = x.iter().map(|&i| i * i).sum();
    let sum_y_sq: f64 = y.iter().map(|&i| i * i).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&i, &j)| i * j).sum();

    let numerator = n as f64 * sum_xy - sum_x * sum_y;
    let denominator =
        ((n as f64 * sum_x_sq - sum_x * sum_x) * (n as f64 * sum_y_sq - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
