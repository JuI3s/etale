# Trace Pairing

Recovers inner products from packed ring elements via the trace form on cyclotomic rings.
Theory: [trace-pairing.pdf](../../math/trace-pairing.pdf).

## API

```rust,ignore
use etale::lattice::pairing::{
    TracePairingParams, pack, inner_product_from_trace,
};
use etale::lattice::trace::GaloisSubgroup;

// Setup parameters (Hachi: d=1024, k=4)
let params = TracePairingParams::hachi();
let h = GaloisSubgroup::new(params.d, params.k);

// Pack vectors into ring elements (n = 256 elements each)
let a: Vec<F> = (0..params.n).map(|i| F::from(i as u64)).collect();
let b: Vec<F> = (0..params.n).map(|i| F::from((i + 1) as u64)).collect();

let packed_a = pack(&a, &params);
let packed_b = pack(&b, &params);

// Recover ⟨a, b⟩ via trace pairing
let inner_product = inner_product_from_trace(&packed_a, &packed_b, &params, &h);
```

> **Note:** `F` is a generic field type implementing `ark_ff::Field`. See tests in
> `src/lattice/pairing.rs` for concrete examples using `Fp64`.

## Parameters

| Parameter | Description | Typical value |
|-----------|-------------|---------------|
| `d` | Ring dimension (power of 2) | 1024 |
| `k` | Extension degree | 4 |
| `n` | Packing capacity: `d/k` | 256 |
| `half_n` | Half packing dimension: `d/(2k)` | 128 |
| `scale` | Inner product scaling factor: `d/k` | 256 |

## Functions

| Function | Description |
|----------|-------------|
| `pack(a, params)` | Maps `F^n → R_q` using basis `{X^i} ∪ {X^{d/2+i}}` |
| `unpack(x, params)` | Inverse of `pack` |
| `trace_pairing(a, b, h)` | Computes `Tr_H(a · σ_{-1}(b))` |
| `inner_product_from_trace(a, b, params, h)` | Recovers `⟨a, b⟩` from packed elements |
| `ring_mul(a, b)` | Ring multiplication in `R_q = Z_q[X]/(X^d + 1)` |

