# Partially Splitting Rings

## Create Splitting Parameters

```rust
use etale::lattice::splitting::SplittingParams;

// Manually specify num_splits
let params = SplittingParams::power_of_two(256, 256, 1, 60, 8380417);

// Or compute num_splits automatically from n and modulus
let params = SplittingParams::with_computed_splits(256, 1, 60, 8380417);
```

## Query Parameters

```rust
let size = params.challenge_set_size();
let bits = params.security_bits();

println!("Challenge set size: {}", size);
println!("Security bits: {}", bits);
```

## Validate Parameters

```rust
// Check if num_splits matches the computed value
assert!(params.is_valid_num_splits());

// Or panic with details if invalid
params.validate();
```

## Compute Number of Splits

```rust
// Compute how X^n + 1 splits mod q
let num_splits = SplittingParams::compute_num_splits(256, 8380417);
```

