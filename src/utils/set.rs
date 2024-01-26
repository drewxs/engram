/// Intersection of two sets.
///
/// # Examples
///
/// ```
/// # use engram::utils::intersection;
/// let a = [true, true, false, false];
/// let b = [true, false, true, false];
/// assert_eq!(intersection(&a, &b), 1);
/// ```
pub fn intersection(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b.iter()).filter(|(&a, &b)| a && b).count()
}

/// Union of two sets.
///
/// # Examples
///
/// ```
/// # use engram::utils::union;
/// let a = [true, true, false, false];
/// let b = [true, false, true, false];
/// assert_eq!(union(&a, &b), 3);
/// ```
pub fn union(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b.iter()).filter(|(&a, &b)| a || b).count()
}
