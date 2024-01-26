use crate::{
    linalg::dot,
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
