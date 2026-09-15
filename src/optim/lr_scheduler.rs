//! Learning rate schedules, like `torch.optim.lr_scheduler`.
//!
//! Schedules are plain functions of the step number. Look up the rate at the start of each step or epoch and pass it to [`Optimizer::set_learning_rate`](crate::optim::Optimizer::set_learning_rate), which leaves the optimizer free to be borrowed elsewhere in the loop.
//!
//! ```
//! # use engram::optim::{lr_scheduler::{CosineAnnealing, Schedule}, Optimizer, Sgd};
//! let schedule = CosineAnnealing::new(0.1, 0.001, 100).with_warmup(5);
//! let mut optimizer = Sgd::new(Vec::new(), schedule.learning_rate(0));
//! for epoch in 0..105 {
//!     optimizer.set_learning_rate(schedule.learning_rate(epoch));
//!     // Train for one epoch.
//! }
//! assert!(optimizer.learning_rate() < 0.002);
//! ```

use std::f32::consts::PI;

/// A learning rate that depends only on the step number.
pub trait Schedule {
    /// Returns the learning rate for `step`, counting from zero.
    fn learning_rate(&self, step: usize) -> f32;

    /// Ramps the rate up linearly over the first `steps` steps before following this schedule.
    fn with_warmup(self, steps: usize) -> Warmup<Self>
    where
        Self: Sized,
    {
        Warmup {
            schedule: self,
            steps,
        }
    }
}

/// Multiplies the rate by `gamma` every `step_size` steps, like `StepLR`.
#[derive(Clone, Copy, Debug)]
pub struct StepLr {
    initial: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLr {
    /// # Panics
    ///
    /// Panics if `step_size` is zero.
    pub fn new(initial: f32, step_size: usize, gamma: f32) -> StepLr {
        assert!(step_size > 0, "StepLr step size must be positive");
        StepLr {
            initial,
            step_size,
            gamma,
        }
    }
}

impl Schedule for StepLr {
    fn learning_rate(&self, step: usize) -> f32 {
        self.initial * self.gamma.powi(exponent(step / self.step_size))
    }
}

/// Multiplies the rate by `gamma` every step, like `ExponentialLR`.
#[derive(Clone, Copy, Debug)]
pub struct ExponentialLr {
    initial: f32,
    gamma: f32,
}

impl ExponentialLr {
    pub fn new(initial: f32, gamma: f32) -> ExponentialLr {
        ExponentialLr { initial, gamma }
    }
}

impl Schedule for ExponentialLr {
    fn learning_rate(&self, step: usize) -> f32 {
        self.initial * self.gamma.powi(exponent(step))
    }
}

/// Follows half a cosine from `initial` down to `min` over `steps` steps and then stays at `min`, like `CosineAnnealingLR`.
#[derive(Clone, Copy, Debug)]
pub struct CosineAnnealing {
    initial: f32,
    min: f32,
    steps: usize,
}

impl CosineAnnealing {
    pub fn new(initial: f32, min: f32, steps: usize) -> CosineAnnealing {
        CosineAnnealing {
            initial,
            min,
            steps,
        }
    }
}

impl Schedule for CosineAnnealing {
    fn learning_rate(&self, step: usize) -> f32 {
        let progress = step.min(self.steps) as f32 / self.steps.max(1) as f32;
        self.min + (self.initial - self.min) * (1.0 + (PI * progress).cos()) / 2.0
    }
}

/// Scales another schedule up linearly to its first rate, then continues with it. Created by [`Schedule::with_warmup`].
#[derive(Clone, Copy, Debug)]
pub struct Warmup<S> {
    schedule: S,
    steps: usize,
}

impl<S: Schedule> Schedule for Warmup<S> {
    fn learning_rate(&self, step: usize) -> f32 {
        if step < self.steps {
            self.schedule.learning_rate(0) * (step + 1) as f32 / self.steps as f32
        } else {
            self.schedule.learning_rate(step - self.steps)
        }
    }
}

/// Lowers the learning rate when a metric stops improving, like `ReduceLROnPlateau` with its default settings.
///
/// Lower values count as improvements, so pass a loss directly or negate a score such as accuracy.
///
/// ```
/// # use engram::optim::lr_scheduler::ReduceOnPlateau;
/// let mut plateau = ReduceOnPlateau::new(0.1).patience(1);
/// assert_eq!(plateau.step(0.5), 0.1);
/// assert_eq!(plateau.step(0.5), 0.1);
/// assert!((plateau.step(0.5) - 0.01).abs() < 1e-8);
/// ```
#[derive(Clone, Debug)]
pub struct ReduceOnPlateau {
    learning_rate: f32,
    factor: f32,
    patience: usize,
    threshold: f32,
    min_learning_rate: f32,
    best: f32,
    stale_steps: usize,
}

impl ReduceOnPlateau {
    pub fn new(learning_rate: f32) -> ReduceOnPlateau {
        ReduceOnPlateau {
            learning_rate,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            min_learning_rate: 0.0,
            best: f32::INFINITY,
            stale_steps: 0,
        }
    }

    /// Sets the multiplier applied to the rate on each reduction.
    pub fn factor(mut self, factor: f32) -> ReduceOnPlateau {
        self.factor = factor;
        self
    }

    /// Sets how many steps without improvement are tolerated before reducing the rate.
    pub fn patience(mut self, patience: usize) -> ReduceOnPlateau {
        self.patience = patience;
        self
    }

    /// Sets the rate below which no further reductions happen.
    pub fn min_learning_rate(mut self, min_learning_rate: f32) -> ReduceOnPlateau {
        self.min_learning_rate = min_learning_rate;
        self
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Records the latest metric and returns the learning rate to use from now on.
    pub fn step(&mut self, metric: f32) -> f32 {
        if metric < self.best * (1.0 - self.threshold) {
            self.best = metric;
            self.stale_steps = 0;
        } else {
            self.stale_steps += 1;
            if self.stale_steps > self.patience {
                self.learning_rate = (self.learning_rate * self.factor).max(self.min_learning_rate);
                self.stale_steps = 0;
            }
        }
        self.learning_rate
    }
}

fn exponent(n: usize) -> i32 {
    n.min(i32::MAX as usize) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rates(schedule: &impl Schedule, steps: usize) -> Vec<f32> {
        (0..steps)
            .map(|step| schedule.learning_rate(step))
            .collect()
    }

    #[test]
    fn decays_in_steps_and_exponentially() {
        assert_eq!(
            rates(&StepLr::new(1.0, 2, 0.5), 5),
            [1.0, 1.0, 0.5, 0.5, 0.25]
        );
        assert_eq!(ExponentialLr::new(1.0, 0.5).learning_rate(3), 0.125);
    }

    #[test]
    fn cosine_annealing_ends_at_minimum() {
        let schedule = CosineAnnealing::new(1.0, 0.1, 10);
        assert_eq!(schedule.learning_rate(0), 1.0);
        assert!((schedule.learning_rate(5) - 0.55).abs() < 1e-6);
        assert!((schedule.learning_rate(10) - 0.1).abs() < 1e-6);
        assert!((schedule.learning_rate(50) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn warmup_ramps_linearly_then_hands_over() {
        let schedule = StepLr::new(1.0, 2, 0.5).with_warmup(4);
        assert_eq!(rates(&schedule, 7), [0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn plateau_waits_for_patience_and_respects_minimum() {
        let mut plateau = ReduceOnPlateau::new(1.0)
            .patience(2)
            .factor(0.5)
            .min_learning_rate(0.4);
        assert_eq!(plateau.step(1.0), 1.0);
        assert_eq!(plateau.step(1.0), 1.0);
        assert_eq!(plateau.step(1.0), 1.0);
        assert_eq!(plateau.step(1.0), 0.5);
        assert_eq!(plateau.step(0.2), 0.5);
        for _ in 0..3 {
            plateau.step(0.3);
        }
        assert_eq!(plateau.learning_rate(), 0.4);
    }
}
