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
/// # use engram::statistics::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mean = mean(&values);
/// # assert_eq!(mean, 3.5);
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
/// # use engram::statistics::*;
/// let values = vec![1.0, 9.0, 2.5, 3.0, 2.0, 8.0];
/// let median = median(&values);
/// # assert_eq!(median, Some(2.75));
/// ```
pub fn median<T: Into<f64> + Copy>(data: &[T]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut sorted = data.iter().map(|&x| x.into()).collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
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
/// # use engram::statistics::*;
/// let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 9];
/// let mode = mode(&values);
/// # assert_eq!(mode, Some(9));
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
/// # use engram::statistics::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let variance = var(&values);
/// # assert_eq!(variance, 2.5);
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
/// # use engram::statistics::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let p_var= pop_var(&values);
/// # assert_eq!(p_var, 2.0);
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
/// # use engram::statistics::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let std_dev = std(&values);
/// # assert_eq!(std_dev, 1.5811388300841898);
/// ```
pub fn std<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    var(data).sqrt()
}

/// Returns the population standard deviation of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::statistics::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let pop_std = pop_std(&values);
/// # assert_eq!(pop_std, 1.4142135623730951);
/// ```
pub fn pop_std<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    pop_var(data).sqrt()
}

/// Returns the sample standard error of a list of values.
///
/// # Examples
///
/// ```
/// # use engram::statistics::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mean_std_err = mean_std_err(&values);
/// # assert_eq!(mean_std_err, 0.7071067811865476);
/// ```
pub fn mean_std_err<T: Into<f64> + Copy + Add + Sub>(data: &[T]) -> f64 {
    std(data) / (data.len() as f64).sqrt()
}

/// Returns a confidence interval for the mean of a list of values, using the normal approximation `mean ± z * standard error`.
///
/// `confidence` is the probability covered by the interval and must lie strictly between 0 and 1. For small samples this interval is narrower than one based on Student's t distribution.
///
/// # Examples
///
/// ```
/// # use engram::statistics::*;
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (low, high) = mean_ci(&values, 0.95).unwrap();
/// # assert!((low - 1.614096175650322).abs() < 1e-6);
/// # assert!((high - 4.385903824349678).abs() < 1e-6);
/// ```
pub fn mean_ci<T: Into<f64> + Copy + Add + Sub>(
    data: &[T],
    confidence: f64,
) -> Result<(f64, f64), String> {
    if data.is_empty() {
        return Err("Empty data".to_string());
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err("Invalid confidence level".to_string());
    }

    let mean = mean(data);
    let margin = normal_quantile(0.5 + confidence / 2.0) * mean_std_err(data);

    Ok((mean - margin, mean + margin))
}

/// Returns the value below which a fraction `p` of the standard normal distribution falls.
///
/// Uses Acklam's rational approximation, whose relative error is below 1.15e-9.
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    const TAIL: f64 = 0.02425;

    let tail = |q: f64| {
        let numerator = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let denominator = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        numerator / denominator
    };
    if p < TAIL {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - TAIL {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        let numerator = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let denominator = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantile_matches_known_values() {
        assert_eq!(normal_quantile(0.5), 0.0);
        assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-8);
        assert!((normal_quantile(0.01) + 2.326347874040841).abs() < 1e-8);
        assert!((normal_quantile(0.999) - 3.090232306167813).abs() < 1e-8);
    }

    #[test]
    fn median_tolerates_nan() {
        assert_eq!(median(&[3.0, f64::NAN, 1.0]), Some(3.0));
    }
}
