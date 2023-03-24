use rand::Rng;

// Gaussian, µ = 0, σ = √[2 / (f_in + f_out)]
pub fn xavier_initialization(f_in: usize, f_out: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut weights = vec![0.0; f_in * f_out];
    for w in weights.iter_mut() {
        *w = rng.gen_range(-1.0..1.0) * (2.0 / (f_in + f_out) as f64).sqrt();
    }
    weights
}

// Gaussian, µ = 0, σ = √[2 / f_in]
pub fn kaiming_initialization(f_in: usize, f_out: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut weights = vec![0.0; f_in * f_out];
    for w in weights.iter_mut() {
        *w = rng.gen_range(-1.0..1.0) * (2.0 / f_in as f64).sqrt();
    }
    weights
}
