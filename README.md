# lattice-crypto

Lattice-based cryptography primitives. More details and references are in my private notes.

Documentation available at https://jui3s.github.io/lattice-crypto/.

> **⚠️ Work in progress.** This library is experimental and under active development—APIs may change without notice. Docs drafted with AI assistance and reviewed by author.

## Tower Trace (32× speedup)

Trace over Galois subgroup $H = \langle \sigma_{-1}, \sigma_{4k+1} \rangle$:

```
Naive:  Tr_H(x) = Σ_{σ∈H} σ(x)              → 256 automorphisms
Tower:  Tr_H(x) = (1+σ_{g₁})...(1+σ_{g₈})(x) → 8 automorphisms
```

Exploits index-2 subgroup tower for 2-groups. Measured 31.7× speedup (d=1024, k=4).

## Rejection Sampling

Rejection sampling for zero-knowledge applications.

## Usage

```bash
cargo test --release
cargo bench
```
