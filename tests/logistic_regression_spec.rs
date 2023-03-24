use engram::logistic_regression::LogisticRegression;

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
