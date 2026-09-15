//! Seedable random number generation.
//!
//! Every thread owns a generator seeded from system entropy. Call [`manual_seed`] to make weight initialization, dropout and shuffling reproducible.

use std::{
    cell::RefCell,
    collections::hash_map::RandomState,
    f64::consts::TAU,
    hash::{BuildHasher, Hasher},
};

thread_local! {
    static RNG: RefCell<Rng> = RefCell::new(Rng::from_entropy());
}

/// Seeds the generator of the current thread.
///
/// ```
/// # use engram::random::{manual_seed, with_rng};
/// manual_seed(42);
/// let a = with_rng(|rng| rng.next_u64());
/// manual_seed(42);
/// assert_eq!(a, with_rng(|rng| rng.next_u64()));
/// ```
pub fn manual_seed(seed: u64) {
    RNG.with(|rng| *rng.borrow_mut() = Rng::new(seed));
}

/// Runs `f` with the generator of the current thread.
pub fn with_rng<T>(f: impl FnOnce(&mut Rng) -> T) -> T {
    RNG.with(|rng| f(&mut rng.borrow_mut()))
}

/// Shuffles a slice in place using the generator of the current thread.
pub fn shuffle<T>(slice: &mut [T]) {
    with_rng(|rng| rng.shuffle(slice));
}

/// A fast, non-cryptographic xoshiro256++ generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    /// Creates a generator from a seed. Equal seeds produce equal sequences on every platform.
    pub fn new(seed: u64) -> Self {
        // Expand the seed with splitmix64 so that similar seeds still give unrelated states.
        let mut x = seed;
        let mut next = move || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Rng {
            state: [next(), next(), next(), next()],
            spare_normal: None,
        }
    }

    fn from_entropy() -> Self {
        Self::new(RandomState::new().build_hasher().finish())
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Returns a uniform sample from `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Returns a uniform sample from `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Returns a uniform sample from `[low, high)`.
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Returns a sample from a normal distribution.
    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        let z = match self.spare_normal.take() {
            Some(z) => z,
            None => {
                // Box-Muller produces two independent samples, so the second one is kept for the next call.
                let u1 = 1.0 - self.next_f64();
                let u2 = self.next_f64();
                let r = (-2.0 * u1.ln()).sqrt();
                self.spare_normal = Some(r * (TAU * u2).sin());
                r * (TAU * u2).cos()
            }
        };
        mean + std * z as f32
    }

    /// Returns a uniform integer from `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Shuffles a slice in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            slice.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_xoshiro256plusplus() {
        let mut rng = Rng {
            state: [1, 2, 3, 4],
            spare_normal: None,
        };
        assert_eq!(rng.next_u64(), 41943041);
        assert_eq!(rng.next_u64(), 58720359);
    }

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let mut c = Rng::new(8);
        let xs: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        let zs: Vec<u64> = (0..16).map(|_| c.next_u64()).collect();
        assert_eq!(xs, ys);
        assert_ne!(xs, zs);
    }

    #[test]
    fn manual_seed_resets_thread_generator() {
        manual_seed(3);
        let a: Vec<f32> = (0..8)
            .map(|_| with_rng(|rng| rng.normal(0.0, 1.0)))
            .collect();
        manual_seed(3);
        let b: Vec<f32> = (0..8)
            .map(|_| with_rng(|rng| rng.normal(0.0, 1.0)))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn uniform_stays_in_range() {
        let mut rng = Rng::new(0);
        for _ in 0..10_000 {
            let x = rng.uniform(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&x));
            assert!(rng.below(5) < 5);
        }
    }

    #[test]
    fn normal_has_expected_moments() {
        let mut rng = Rng::new(0);
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal(1.5, 2.0) as f64).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((mean - 1.5).abs() < 0.03, "mean {mean}");
        assert!((var.sqrt() - 2.0).abs() < 0.03, "std {}", var.sqrt());
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(1);
        let mut values: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut values);
        assert_ne!(values, (0..100).collect::<Vec<_>>());
        values.sort();
        assert_eq!(values, (0..100).collect::<Vec<_>>());
    }
}
