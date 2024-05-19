/// Intersection of two sets.
///
/// # Examples
///
/// ```
/// # use engram::utils::intersection;
/// let a = [true, true, false, false];
/// let b = [true, false, true, false];
/// let i = intersection(&a, &b);
/// # assert_eq!(i, 1);
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
/// let u = union(&a, &b);
/// # assert_eq!(u, 3);
/// ```
pub fn union(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b.iter()).filter(|(&a, &b)| a || b).count()
}
