# Trace Computation

## Create a Ring Element

```rust
use etale::lattice::trace::CyclotomicRingElement;

let coeffs = vec![1, 2, 3, 0, 0, 0, 0, 0];
let x = CyclotomicRingElement::new(coeffs, 8);
```

## Define a Galois Subgroup

```rust
use etale::lattice::trace::GaloisSubgroup;

let n = 256;
let k = 1;
let subgroup = GaloisSubgroup::standard_subgroup(n, k);

println!("Order: {}", subgroup.order());
```

## Compute the Trace

```rust
use etale::lattice::trace::TraceComputer;

let computer = TraceComputer::new(n);
let subgroup = GaloisSubgroup::standard_subgroup(n, 1);

let trace = computer.compute_trace(&x, &subgroup);
```

## Tower Optimization

```rust
// Faster trace computation using tower decomposition
if let Some(tower_gens) = subgroup.tower_generators() {
    let trace = computer.trace_tower(&x, &tower_gens);
}
```

