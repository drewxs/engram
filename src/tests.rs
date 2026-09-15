//! End-to-end training runs across the tensor, nn, optim and data modules.

use crate::{
    data::{DataLoader, TensorDataset},
    manual_seed,
    nn::{self, functional as F, Module},
    no_grad,
    optim::{Adam, Optimizer, Sgd},
    tensor, Tensor,
};

#[test]
fn xor_converges_for_every_seed() {
    let x = tensor![[0, 0], [0, 1], [1, 0], [1, 1]];
    let y = tensor![[0], [1], [1], [0]];
    for seed in 0..20 {
        manual_seed(seed);
        let model = nn::Sequential::new()
            .add(nn::Linear::new(2, 8))
            .add(nn::Tanh)
            .add(nn::Linear::new(8, 1));
        let mut opt = Adam::new(model.parameters(), 0.05);
        for _ in 0..300 {
            let loss = F::binary_cross_entropy_with_logits(&model.forward(&x), &y);
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
        let predictions = no_grad(|| model.forward(&x)).greater(0.0);
        assert_eq!(predictions, y, "seed {seed}");
    }
}

#[test]
fn linear_regression_recovers_weights_with_sgd() {
    manual_seed(1);
    let x = Tensor::randn([512, 2]);
    let noise = Tensor::normal([512, 1], 0.0, 0.01);
    let y = x.matmul(&tensor![[3], [-2]]) + 1.0 + noise;
    let loader = DataLoader::new(TensorDataset::new(x, y), 32).shuffle(true);

    let model = nn::Linear::new(2, 1);
    let mut opt = Sgd::new(model.parameters(), 0.05).momentum(0.9);
    for _ in 0..20 {
        for (inputs, targets) in &loader {
            let loss = F::mse_loss(&model.forward(&inputs), &targets);
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
    }
    assert!(
        model.weight.allclose(&tensor![[3, -2]], 0.02),
        "{}",
        model.weight
    );
    assert!(model.bias.unwrap().allclose(&tensor![1], 0.02));
}

#[test]
fn classifies_gaussian_blobs() {
    manual_seed(2);
    let centers = tensor![[-2, 0], [2, 0], [0, 2.5]];
    let labels: Vec<usize> = (0..600).map(|i| i % 3).collect();
    let x = centers.index_select(0, &labels) + Tensor::normal([600, 2], 0.0, 0.5);
    let y = Tensor::from_vec(labels.iter().map(|&l| l as f32).collect(), [600]);
    let loader = DataLoader::new(TensorDataset::new(x.clone(), y.clone()), 32).shuffle(true);

    let mut model = nn::Sequential::new()
        .add(nn::Linear::new(2, 32))
        .add(nn::Relu)
        .add(nn::Dropout::new(0.1))
        .add(nn::Linear::new(32, 3));
    let mut opt = Adam::new(model.parameters(), 0.01).decoupled_weight_decay(1e-4);
    for _ in 0..15 {
        for (inputs, targets) in &loader {
            let loss = F::cross_entropy(&model.forward(&inputs), &targets);
            opt.zero_grad();
            loss.backward();
            nn::utils::clip_grad_norm(opt.parameters(), 5.0);
            opt.step();
        }
    }

    model.eval();
    let accuracy = no_grad(|| model.forward(&x))
        .argmax(1, false)
        .equal(y)
        .mean()
        .item();
    assert!(accuracy > 0.95, "accuracy {accuracy}");
}
