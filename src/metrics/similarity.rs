use super::norm_l2;

/// Compute the cosine similarity between two vectors.
///
/// # Examples
///
/// ```
/// # use engram::metrics::cosine_similarity;
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.0, 2.0, 3.0];
/// let c = [1.0, 2.0, 4.0];
/// assert_eq!(cosine_similarity(&a, &b), 1.0);
/// assert_eq!(cosine_similarity(&a, &c), 0.9914601339836675);
/// ```
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b.iter()).fold(0.0, |acc, (x, y)| acc + x * y);
    let norm_a = norm_l2(a);
    let norm_b = norm_l2(b);
    dot / (norm_a * norm_b)
}
