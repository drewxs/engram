//! Logistic regression model.

use rand::Rng;

pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub learning_rate: f64,
    pub num_iterations: usize,
    pub regularization_param: f64,
}

impl LogisticRegression {
    pub fn new(num_features: usize) -> LogisticRegression {
        let mut rng = rand::thread_rng();
        let mut weights = vec![0.0; num_features];
        for w in weights.iter_mut() {
            *w = rng.gen_range(-0.1..0.1);
        }
        LogisticRegression {
            weights,
            learning_rate: 0.01,
            num_iterations: 1000, // TODO: implement AdaGrad, RMSProp, Adam, etc.
            regularization_param: 1.0,
        }
    }

    pub fn train(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");

        let m = x.len();
        let n = x[0].len();

        for i in 0..self.num_iterations {
            let (grad_sum, error_sum) = self.gradient_descent(x, y, m, n);
            let error_avg = error_sum / (m as f64);
            for j in 0..n {
                let grad = (grad_sum[j] / (m as f64))
                    + ((self.regularization_param * self.weights[j]) / (m as f64));
                self.weights[j] -= self.learning_rate * grad;
            }

            // Early stopping based on the average error
            if i > 0 && error_avg > 0.0 && error_avg > self.error_avg_last_iteration(x, y) {
                break;
            }
        }
    }

    pub fn gradient_descent(
        &self,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        m: usize,
        n: usize,
    ) -> (Vec<f64>, f64) {
        let mut error_sum = 0.0;
        let mut grad_sum = vec![0.0; n];

        for i in 0..m {
            let h = self.predict(&x[i]);
            let error = h - y[i];
            error_sum += error;

            for j in 0..n {
                grad_sum[j] += error * x[i][j];
            }
        }

        (grad_sum, error_sum)
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        let z: f64 = self.weights.iter().zip(x.iter()).map(|(w, x)| w * x).sum();
        1.0 / (1.0 + (-z).exp())
    }

    fn error_avg_last_iteration(&self, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> f64 {
        let m = x.len();
        let mut error_sum = 0.0;
        for i in 0..m {
            let h = self.predict(&x[i]);
            let error = h - y[i];
            error_sum += error;
        }
        error_sum / (m as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression() {
        // Input data/labels
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![-1.0, -1.0],
            vec![-2.0, -1.0],
            vec![-1.5, 0.5],
            vec![1.5, 1.5],
        ];
        let y = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // Train model on input data
        let mut lr = LogisticRegression::new(2);
        lr.learning_rate = 0.01;
        lr.num_iterations = 5000;
        lr.regularization_param = 1.0;
        lr.train(&x, &y);

        // Test trained model on input data
        let test_data = vec![vec![0.5, 0.5], vec![-1.0, 1.0], vec![2.0, -2.0]];
        let expected_predictions = vec![0.65, 0.42, 0.65];
        let epsilon = 0.1;

        for (i, d) in test_data.iter().enumerate() {
            let prediction = lr.predict(d);
            let expected = expected_predictions[i];
            println!("prediction: {}, expected: {}", prediction, expected);
            assert!((prediction - expected).abs() < epsilon);
        }
    }
}
