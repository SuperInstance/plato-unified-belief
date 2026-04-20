# plato-unified-belief

Three Bayesian belief systems in one score: **confidence** (sensor fusion), **trust** (agent reliability), and **relevance** (knowledge weight).

## Why

Every PLATO agent needs to answer three questions about every piece of information:
- How confident am I? (sensor fusion from `flux-confidence`)
- How reliable is the source? (agent trust from `cuda-trust`)
- How relevant is this right now? (tile weight from `plato-tiling`)

Same math (Bayesian update + temporal decay), three purposes. This crate unifies them.

## Usage

```rust
use plato_unified_belief::{BeliefScore, BeliefDimension};

let mut score = BeliefScore::new();
score.update(BeliefDimension::Confidence, 0.8);
score.update(BeliefDimension::Trust, 0.6);
score.decay(0.95); // temporal decay per tick
```

Zero dependencies. `cargo add plato-unified-belief`
