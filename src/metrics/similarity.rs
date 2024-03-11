use crate::{
    linalg::{dot, magnitude},
    utils::{intersection, union},
};

use super::norm_l2;

/// Compute the cosine similarity between two vectors.
/// Formula: dot(a, b) / (|a| * |b|)
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
    dot(a, b) / (norm_l2(a) * norm_l2(b))
}

/// Compute the Jaccard index between two sets.
/// Formula: |A ∩ B| / |A ∪ B|
///
/// # Examples
///
/// ```
/// # use engram::metrics::jaccard_index;
/// let a = [true, true, false, false];
/// let b = [true, false, true, false];
/// let c = [false, false, false, false];
/// assert_eq!(jaccard_index(&a, &b), 0.3333333333333333);
/// assert_eq!(jaccard_index(&a, &c), 0.0);
/// ```
pub fn jaccard_index(a: &[bool], b: &[bool]) -> f64 {
    let intersection = intersection(a, b);
    let union = union(a, b);
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute the Tanimoto Coefficient for continuous vectors.
/// A measure of similarity for real-valued vectors that generalizes the Jaccard index.
/// Formula: dot(a, b) / (|a|^2 + |b|^2 - dot(a, b))
///
/// # Examples
///
/// ```
/// # use engram::metrics::tanimoto_coefficient;
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.3, 2.1, 3.0];
/// let c = [1.5, 2.4, 4.0];
/// assert!((tanimoto_coefficient(&a, &b) - 0.9932).abs() < 1e-4);
/// assert!((tanimoto_coefficient(&a, &c) - 0.9285).abs() < 1e-4);
/// ```
pub fn tanimoto_coefficient(a: &[f64], b: &[f64]) -> f64 {
    let dot = dot(a, b);
    let denominator = magnitude(a).powi(2) + magnitude(b).powi(2) - dot;
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}
