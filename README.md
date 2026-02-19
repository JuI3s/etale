# Étale

Lattice-based cryptography primitives. Every security claim is proved and cited. No folklore.

## Background

- **[Trace pairing](math/trace-pairing.pdf)** — inner product recovery from the trace form on finite étale algebras, specializing to the orthogonla basis in 2-power cyclotomic rings.
- **[Tower trace computation](math/galois-tower.pdf)** — logarithmic-time field trace via transitivity over a tower of quadratic extensions. ~32× speedup for practical parameters.

## Features

### Ring Arithmetic
NTT-based and schoolbook multiplication for cyclotomic rings R_q = Z_q[X]/(X^d + 1), sparse ternary multiplication for challenge polynomials, base-b decomposition and recomposition.

### Ajtai Commitment
Generic SIS-based commitment t = A·s with optional base-b decomposition for polynomial commitment schemes.

### Galois Tower Trace (32× speedup)
Trace over Galois subgroup via index-2 tower decomposition. Measured 31.7× speedup (d=1024, k=4).

### Parameter Database
Hachi, Greyhound, compressed tiers (k=4,8,16,32), Dilithium, Falcon, Kyber.

### Benchmarks
Comprehensive criterion benchmarks: ring multiplication (schoolbook vs NTT), sparse multiplication, NTT transforms, decomposition, and commitment across all parameter sets.

## Documentation

🚧 Under construction — https://jui3s.github.io/etale/

> **⚠️ Work in progress.** Experimental, APIs may change.

## Usage
```bash
cargo test --release
cargo bench
```