//! Statistics.

use std::{
    collections::HashMap,
    hash::Hash,
    ops::{Add, Sub},
};

/// Returns the mean of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// assert_eq!(mean(&values), 3.5);
/// ```
pub fn mean<T: Into<f64> + Copy>(data: &[T]) -> f64 {
    let mut sum = 0.0;
    for &x in data {
        sum += x.into();
    }
    sum / data.len() as f64
}

/// Returns the median of a list of values
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 9.0, 2.5, 3.0, 2.0, 8.0];
/// assert_eq!(median(&values), Some(2.75));
/// ```
pub fn median<T: Into<f64> + Copy>(data: &[T]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut sorted = data.iter().map(|&x| x.into()).collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;

    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

/// Returns the mode of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 9];
/// assert_eq!(mode(&values), Some(9));
/// ```
pub fn mode<T: Copy + Eq + Hash>(data: &[T]) -> Option<T> {
    if data.is_empty() {
        return None;
    }

    let mut counts: HashMap<T, i64> = HashMap::new();
    data.iter().copied().max_by_key(|&x| {
        let count = counts.entry(x).or_insert(0);
        *count += 1;
        *count
    })
}

/// Returns the sample variance of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(var(&values), 2.5);
/// ```
pub fn var<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    let mean = mean(data);
    data.iter()
        .map(|&x| ((x.into() - mean).powi(2)) / (data.len() - 1) as f64)
        .sum()
}

/// Returns the population variance of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(pop_var(&values), 2.0);
/// ```
pub fn pop_var<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    let mean = mean(data);
    data.iter()
        .map(|&x| ((x.into() - mean).powi(2)) / data.len() as f64)
        .sum()
}

/// Returns the sample standard deviation of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(std(&values), 1.5811388300841898);
/// ```
pub fn std<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    var(data).sqrt()
}

/// Returns the population standard deviation of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(pop_std(&values), 1.4142135623730951);
/// ```
pub fn pop_std<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    pop_var(data).sqrt()
}

/// Returns the sample standard error of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(mean_std_err(&values), 0.7071067811865476);
/// ```
pub fn mean_std_err<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    std(data) / (data.len() as f64).sqrt()
}

/// Returns the mean confidence interval of a list of values.
/// Confidence level is a value between 0 and 1.
///
/// # Examples
///
/// ```
/// # use engram::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(mean_ci(&values, 0.95).unwrap(), (2.32824855787278, 3.67175144212722));
/// ```
#[derive(Debug)]
pub enum MeanCIError {
    EmptyData,
    InvalidConfidence,
}
pub fn mean_ci<T: Into<f64> + Copy + Add + Sub>(
    data: &[T],
    confidence: f64,
) -> Result<(f64, f64), MeanCIError> {
    if data.is_empty() {
        return Err(MeanCIError::EmptyData);
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(MeanCIError::InvalidConfidence);
    }

    let mean = mean(data);
    let mean_std_err = mean_std_err(data);
    let z_std_err = confidence * mean_std_err;

    Ok((mean - z_std_err, mean + z_std_err))
}
