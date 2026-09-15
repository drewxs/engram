# Engram

General purpose machine learning library for Rust, with an API modeled on PyTorch.

[Documentation](https://engram-rs.vercel.app/engram/index.html)

- N-dimensional tensors with broadcasting and reverse-mode automatic differentiation
- Layers, activations, losses and weight initialization in `nn`, with a functional API in `nn::functional`
- `Sgd` and `Adam` optimizers with momentum, weight decay and gradient clipping
- `Dataset` and `DataLoader` for batching and shuffling
- Seedable random number generation for reproducible runs

## Installation

```sh
cargo add --git https://github.com/drewxs/engram
```

Or add directly to your `Cargo.toml` dependencies.

```toml
engram = { git = "https://github.com/drewxs/engram", version = "0.1.0" }
```

## Example

```rust
use engram::prelude::*;

manual_seed(0);

let x = tensor![[0, 0], [0, 1], [1, 0], [1, 1]];
let y = tensor![[0], [1], [1], [0]];

let model = nn::Sequential::new()
    .add(nn::Linear::new(2, 8))
    .add(nn::Tanh)
    .add(nn::Linear::new(8, 1));
let mut optimizer = Adam::new(model.parameters(), 0.05);

for _ in 0..300 {
    let loss = F::binary_cross_entropy_with_logits(&model.forward(&x), &y);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}

let predictions = no_grad(|| model.forward(&x)).greater(0.0);
assert_eq!(predictions, y);
```

See [`examples`](https://github.com/drewxs/engram/tree/main/examples) for complete programs, such as `cargo run --release --example spirals`.

---

[MIT](https://github.com/drewxs/engram/blob/main/LICENSE) licensed.
