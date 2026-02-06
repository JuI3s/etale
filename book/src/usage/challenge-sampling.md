# Challenge Sampling

## Sample a Random Challenge

```rust
use etale::lattice::splitting::sample_challenge;
use rand::thread_rng;

let mut rng = thread_rng();

// n=256, τ=1 (ternary coefficients), ω=60 (Hamming weight)
let challenge = sample_challenge(&mut rng, 256, 1, 60);

println!("L1 norm: {}", challenge.l1_norm());
println!("Weight: {}", challenge.weight());
```

## Deterministic Challenge from Seed

```rust
use etale::lattice::splitting::challenge_from_seed;

let seed = b"my-seed-bytes";
let challenge = challenge_from_seed(seed, 256, 1, 60);
```

## Challenge Arithmetic

```rust
let c1 = sample_challenge(&mut rng, 256, 1, 60);
let c2 = sample_challenge(&mut rng, 256, 1, 60);

let diff = c1.sub(&c2);
let sum = c1.add(&c2);
```

## Build a Challenge Set

```rust
use etale::lattice::splitting::{ChallengeSet, SplittingParams};

let params = SplittingParams::with_computed_splits(256, 1, 60, 8380417);
let challenge_set = ChallengeSet::build(params, 100);

println!("Set size: {}", challenge_set.size());
```

