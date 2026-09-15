//! Trains a multilayer perceptron to separate three interleaved spirals.
//!
//! Run with `cargo run --release --example spirals`.

use std::f32::consts::TAU;

use engram::{prelude::*, random};

/// Generates `points` noisy samples per class, arranged as `classes` spiral arms.
fn spirals(points: usize, classes: usize) -> (Tensor, Tensor) {
    let mut inputs = Vec::with_capacity(points * classes * 2);
    let mut labels = Vec::with_capacity(points * classes);
    for class in 0..classes {
        for i in 0..points {
            let radius = i as f32 / points as f32;
            let noise = random::with_rng(|rng| rng.normal(0.0, 0.2));
            let angle = class as f32 * TAU / classes as f32 + radius * 4.0 + noise;
            inputs.extend([radius * angle.sin(), radius * angle.cos()]);
            labels.push(class as f32);
        }
    }
    let n = labels.len();
    (
        Tensor::from_vec(inputs, [n, 2]),
        Tensor::from_vec(labels, [n]),
    )
}

fn main() {
    manual_seed(0);
    let (inputs, labels) = spirals(200, 3);
    let dataset = TensorDataset::new(inputs.clone(), labels.clone());
    let loader = DataLoader::new(dataset, 32).shuffle(true);

    let mut model = nn::Sequential::new()
        .add(nn::Linear::new(2, 64))
        .add(nn::Relu)
        .add(nn::Linear::new(64, 64))
        .add(nn::Relu)
        .add(nn::Linear::new(64, 3));
    let mut optimizer = Adam::new(model.parameters(), 0.01);

    for epoch in 1..=100 {
        model.train();
        let mut total_loss = 0.0;
        for (x, y) in &loader {
            let loss = F::cross_entropy(&model.forward(&x), &y);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            total_loss += loss.item();
        }

        if epoch % 10 == 0 {
            model.eval();
            let predictions = no_grad(|| model.forward(&inputs)).argmax(1, false);
            let accuracy = predictions.equal(&labels).mean().item();
            println!(
                "epoch {epoch:>3}  loss {:.4}  accuracy {:.1}%",
                total_loss / loader.len() as f32,
                accuracy * 100.0
            );
        }
    }
}
