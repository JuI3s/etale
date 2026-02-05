# lattice-crypto

Lattice-based cryptography primitives. Every security claim in this library is proved and cited. No folklore.

## Background

- Conceptual framework and optimization — [math/](math/)
  - **[Trace pairing](math/trace-pairing.pdf)** — inner product recovery from the trace form on finite étale algebras, specializing to the self-dual basis in 2-power cyclotomic rings used by [Hachi](https://eprint.iacr.org/2026/156).
  - **[Tower trace computation](math/galois-tower.pdf)** — logarithmic-time field trace via transitivity over a tower of quadratic extensions. ~32× speedup for practical parameters.

## Features

### Galois Tower Trace (32× speedup)

Trace over Galois subgroup $H = \langle \sigma_{-1}, \sigma_{4k+1} \rangle$:

```
Naive:  Tr_H(x) = Σ_{σ∈H} σ(x)              → 256 automorphisms
Tower:  Tr_H(x) = (1+σ_{g₁})...(1+σ_{g₈})(x) → 8 automorphisms
```

Exploits index-2 subgroup tower for 2-groups. Measured 31.7× speedup (d=1024, k=4).

### Rejection Sampling

Rejection sampling for zero-knowledge applications.

## Documentation

Documentation available at https://jui3s.github.io/lattice-crypto/.

> **⚠️ Work in progress.** This library is experimental and under active development—APIs may change without notice. 

## Usage

```bash
cargo test --release
cargo bench
```
