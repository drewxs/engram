//! Rough timings for the hot paths of a training loop.
//!
//! Run with `cargo run --release --example benchmark`, optionally followed by `-- <filter>` to run only the benchmarks whose names contain the filter.

use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use engram::prelude::*;

struct Bench {
    filter: Option<String>,
}

impl Bench {
    /// Times `f` over about half a second after a short warmup and prints the mean time per call.
    fn run(&self, name: &str, mut f: impl FnMut()) {
        if self
            .filter
            .as_ref()
            .is_some_and(|filter| !name.contains(filter.as_str()))
        {
            return;
        }
        for _ in 0..3 {
            f();
        }
        let start = Instant::now();
        let mut iterations = 0;
        while start.elapsed() < Duration::from_millis(500) {
            f();
            iterations += 1;
        }
        let mean = start.elapsed() / iterations;
        println!("{name:<48} {:>9.3} ms", mean.as_secs_f64() * 1e3);
    }
}

fn main() {
    manual_seed(0);
    let bench = Bench {
        filter: env::args().nth(1),
    };

    let a = Tensor::randn([512, 512]);
    let b = Tensor::randn([512, 512]);
    bench.run("matmul [512, 512]", || drop(black_box(a.matmul(&b))));

    let x = Tensor::randn([256, 1024]);
    let bias = Tensor::randn([1024]);
    bench.run("add broadcast [256, 1024] + [1024]", || {
        drop(black_box(&x + &bias))
    });
    bench.run("mul [256, 1024]", || drop(black_box(&x * &x)));
    bench.run("mul scalar [256, 1024]", || drop(black_box(&x * 2.0)));
    bench.run("relu [256, 1024]", || drop(black_box(x.relu())));
    bench.run("softmax [256, 1024]", || drop(black_box(x.softmax(-1))));
    bench.run("softmax over dim 0 [256, 1024]", || {
        drop(black_box(x.softmax(0)))
    });
    bench.run("sum [256, 1024]", || drop(black_box(x.sum())));
    bench.run("sum_dim [256, 1024]", || {
        drop(black_box(x.sum_dim(0, false)))
    });
    bench.run("transpose [256, 1024]", || drop(black_box(x.t())));

    let w = x.set_requires_grad(true);
    bench.run("relu and sum, forward [256, 1024]", || {
        drop(black_box(w.relu().sum()))
    });
    bench.run("relu and sum, forward and backward [256, 1024]", || {
        w.zero_grad();
        w.relu().sum().backward();
    });
    bench.run("softmax and sum, forward and backward [256, 1024]", || {
        w.zero_grad();
        w.softmax(-1).sum().backward();
    });

    let model = nn::Sequential::new()
        .add(nn::Linear::new(784, 512))
        .add(nn::Relu)
        .add(nn::Linear::new(512, 256))
        .add(nn::Relu)
        .add(nn::Linear::new(256, 10));
    let mut optimizer = Adam::new(model.parameters(), 1e-3);
    let inputs = Tensor::randn([128, 784]);
    let targets = Tensor::from_vec((0..128).map(|i| (i % 10) as f32).collect(), [128]);
    bench.run("mlp 784-512-256-10 forward, batch 128", || {
        drop(black_box(no_grad(|| model.forward(&inputs))))
    });
    bench.run("mlp 784-512-256-10 train step, batch 128", || {
        let loss = F::cross_entropy(&model.forward(&inputs), &targets);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    });

    let cnn = nn::Sequential::new()
        .add(nn::Conv2d::new(3, 32, 3).padding(1))
        .add(nn::Relu)
        .add(nn::MaxPool2d::new(2))
        .add(nn::Conv2d::new(32, 64, 3).padding(1))
        .add(nn::Relu)
        .add(nn::MaxPool2d::new(2))
        .add(nn::Flatten::default())
        .add(nn::Linear::new(64 * 8 * 8, 10));
    let mut cnn_optimizer = Adam::new(cnn.parameters(), 1e-3);
    let images = Tensor::randn([32, 3, 32, 32]);
    let labels = Tensor::from_vec((0..32).map(|i| (i % 10) as f32).collect(), [32]);
    bench.run("cnn forward, batch 32 of 3x32x32", || {
        drop(black_box(no_grad(|| cnn.forward(&images))))
    });
    bench.run("cnn train step, batch 32 of 3x32x32", || {
        let loss = F::cross_entropy(&cnn.forward(&images), &labels);
        cnn_optimizer.zero_grad();
        loss.backward();
        cnn_optimizer.step();
    });
}
