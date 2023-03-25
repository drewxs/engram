use rand::Rng;

// Gaussian, µ = 0, σ = √[2 / (f_in + f_out)]
pub fn xavier(f_in: usize, f_out: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut weights = vec![0.0; f_in * f_out];
    for w in weights.iter_mut() {
        *w = rng.gen_range(-1.0..1.0) * (2.0 / (f_in + f_out) as f64).sqrt();
    }
    weights
}

// Gaussian, µ = 0, σ = √[2 / f_in]
pub fn kaiming(f_in: usize, f_out: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut weights = vec![0.0; f_in * f_out];
    for w in weights.iter_mut() {
        *w = rng.gen_range(-1.0..1.0) * (2.0 / f_in as f64).sqrt();
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier() {
        let weights = xavier(2, 3);
        assert_eq!(weights.len(), 6);
        println!("{:?}", weights);
        assert!(weights.iter().all(|w| w.abs() <= 1.0));
    }

    #[test]
    fn test_kaiming() {
        let weights = kaiming(2, 3);
        assert_eq!(weights.len(), 6);
        println!("{:?}", weights);
        assert!(weights.iter().all(|w| w.abs() <= 1.0));
    }
}
