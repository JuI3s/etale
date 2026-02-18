//! NTT-based ring arithmetic for lattice-based polynomial commitments
//!
//! This module provides Number Theoretic Transform (NTT) operations for cyclotomic
//! rings R_q = Z_q[X]/(X^d + 1), supporting both NTT-friendly primes (Kyber, Dilithium)
//! and non-NTT-friendly primes via schoolbook multiplication (Hachi approach).
//!
//! # Mathematical Background
//!
//! For R_q = Z_q[X]/(X^d + 1) where d is a power of 2:
//! - X^d + 1 is the 2d-th cyclotomic polynomial
//! - Its roots are primitive 2d-th roots of unity: ω^{2j+1} for j = 0,...,d-1
//!
//! **NTT-friendly primes:** q ≡ 1 (mod 2d) allows direct negacyclic NTT
//! **Non-NTT-friendly primes:** Use schoolbook or CRT-based multiplication
//!
//! # References
//!
//! - [Hachi](https://eprint.iacr.org/2026/156) - Efficient Lattice-Based Multilinear
//!   Polynomial Commitments over Extension Fields (DQZZ26)
//! - Greyhound (CRYPTO 2024) - NS24
//! - NIST FIPS 203/204/206 for ML-KEM, ML-DSA parameters

use std::fmt;

use super::modular::{add_mod, mod_inv, mul_mod, pow_mod, sub_mod};

// Re-export params for convenience
pub use super::params::{
    find_primitive_2d_root, RingParams, ALL_PARAMS, COMPRESSED_K16, COMPRESSED_K32, COMPRESSED_K4,
    COMPRESSED_K8, DILITHIUM_2, FALCON_512, GREYHOUND, HACHI, HACHI_FAMILY, KYBER_512,
    NTT_FRIENDLY,
};

// ============================================================================
// Ring Element
// ============================================================================

/// Element in cyclotomic ring R_q = Z_q[X]/(X^d + 1) optimized for NTT.
///
/// Coefficients are stored as unsigned values in [0, q).
/// For signed interpretation, use `centered()`.
///
/// # Representation Choice: `Vec<u64>` vs `[u64; D]`
///
/// We use `Vec<u64>` for runtime flexibility across different parameter sets:
///
/// **Why not const generics `[u64; D]`?**
/// - Would require `RingElement<256>`, `RingElement<512>`, etc. as distinct types
/// - Functions would need generic bounds: `fn ntt<const D: usize>(elem: RingElement<D>)`
/// - Benchmark code iterating over parameter sets becomes significantly more complex
///
/// **Performance impact:**
/// - Vec overhead: 1 heap alloc + pointer indirection per element
/// - For d=256..2048 and NTT O(d log d), this is negligible (~0.1% overhead)
/// - Real bottleneck is modular multiplication, not memory layout
///
/// **When to switch to arrays:**
/// - Production code with single fixed d (e.g., Kyber's d=256)
/// - Embedded/no-std environments
/// - When profiling shows memory allocation as bottleneck
#[derive(Clone, PartialEq, Eq)]
pub struct RingElement {
    /// Coefficients: c_0 + c_1·X + ... + c_{d-1}·X^{d-1}
    pub coeffs: Vec<u64>,
    /// Ring dimension d (must be power of 2 for NTT)
    pub d: usize,
    /// Modulus q (should satisfy q ≡ 1 (mod 2d) for direct NTT)
    pub q: u64,
}

impl fmt::Debug for RingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingElement {{ d: {}, q: {}, coeffs: [...] }}", self.d, self.q)
    }
}

impl RingElement {
    /// Create new ring element from coefficients
    pub fn new(coeffs: Vec<u64>, d: usize, q: u64) -> Self {
        assert!(d.is_power_of_two(), "d must be power of two");
        let mut c = coeffs;
        c.resize(d, 0);
        // Reduce coefficients mod q
        for coeff in c.iter_mut() {
            *coeff %= q;
        }
        Self { coeffs: c, d, q }
    }

    /// Create new ring element from signed coefficients
    pub fn from_signed(coeffs: Vec<i64>, d: usize, q: u64) -> Self {
        let unsigned: Vec<u64> = coeffs
            .into_iter()
            .map(|c| ((c % q as i64) + q as i64) as u64 % q)
            .collect();
        Self::new(unsigned, d, q)
    }

    /// Create zero element
    pub fn zero(d: usize, q: u64) -> Self {
        Self {
            coeffs: vec![0; d],
            d,
            q,
        }
    }

    /// Create element representing constant c
    pub fn constant(c: u64, d: usize, q: u64) -> Self {
        let mut coeffs = vec![0; d];
        coeffs[0] = c % q;
        Self { coeffs, d, q }
    }

    /// Create random element
    pub fn random<R: rand::Rng>(rng: &mut R, d: usize, q: u64) -> Self {
        let coeffs: Vec<u64> = (0..d).map(|_| rng.gen_range(0..q)).collect();
        Self { coeffs, d, q }
    }

    /// Create random element with coefficients bounded by `bound`
    pub fn random_bounded<R: rand::Rng>(rng: &mut R, d: usize, q: u64, bound: u64) -> Self {
        let coeffs: Vec<u64> = (0..d).map(|_| rng.gen_range(0..bound)).collect();
        Self::new(coeffs, d, q)
    }

    /// Create sparse random element with `num_nonzero` entries in {-1, 0, 1}
    pub fn random_sparse<R: rand::Rng>(rng: &mut R, d: usize, q: u64, num_nonzero: usize) -> Self {
        use rand::seq::SliceRandom;
        let mut coeffs = vec![0i64; d];
        let mut indices: Vec<usize> = (0..d).collect();
        indices.shuffle(rng);
        for &idx in indices.iter().take(num_nonzero) {
            coeffs[idx] = if rng.gen_bool(0.5) { 1 } else { -1 };
        }
        Self::from_signed(coeffs, d, q)
    }

    /// Get centered representation (coefficients in [-q/2, q/2))
    pub fn centered(&self) -> Vec<i64> {
        let half_q = self.q as i64 / 2;
        self.coeffs
            .iter()
            .map(|&c| {
                let c = c as i64;
                if c > half_q { c - self.q as i64 } else { c }
            })
            .collect()
    }

    /// Compute ℓ∞ norm of centered representation
    pub fn linf_norm(&self) -> u64 {
        self.centered().iter().map(|c| c.abs() as u64).max().unwrap_or(0)
    }

    /// Add two ring elements
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.d, other.d);
        assert_eq!(self.q, other.q);
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| add_mod(a, b, self.q))
            .collect();
        Self {
            coeffs,
            d: self.d,
            q: self.q,
        }
    }

    /// Subtract two ring elements
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.d, other.d);
        assert_eq!(self.q, other.q);
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| sub_mod(a, b, self.q))
            .collect();
        Self {
            coeffs,
            d: self.d,
            q: self.q,
        }
    }

    /// Negate ring element
    pub fn neg(&self) -> Self {
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .map(|&c| if c == 0 { 0 } else { self.q - c })
            .collect();
        Self {
            coeffs,
            d: self.d,
            q: self.q,
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .map(|&c| mul_mod(c, scalar, self.q))
            .collect();
        Self {
            coeffs,
            d: self.d,
            q: self.q,
        }
    }

    /// Schoolbook polynomial multiplication in R_q = Z_q[X]/(X^d + 1)
    ///
    /// Time complexity: O(d²)
    pub fn mul_schoolbook(&self, other: &Self) -> Self {
        assert_eq!(self.d, other.d);
        assert_eq!(self.q, other.q);
        let d = self.d;
        let q = self.q;

        let mut result = vec![0u64; d];

        for i in 0..d {
            if self.coeffs[i] == 0 {
                continue;
            }
            for j in 0..d {
                if other.coeffs[j] == 0 {
                    continue;
                }
                let k = i + j;
                let prod = mul_mod(self.coeffs[i], other.coeffs[j], q);
                if k < d {
                    result[k] = add_mod(result[k], prod, q);
                } else {
                    // X^{d+r} ≡ -X^r mod (X^d + 1)
                    result[k - d] = sub_mod(result[k - d], prod, q);
                }
            }
        }

        Self { coeffs: result, d, q }
    }

    /// Apply automorphism σ_k: X ↦ X^k
    ///
    /// Requires: k odd, gcd(k, 2d) = 1
    ///
    /// For X^j ↦ X^{kj mod 2d}, with sign from reduction mod (X^d + 1)
    pub fn automorphism(&self, k: usize) -> Self {
        let d = self.d;
        let two_d = 2 * d;
        let mut result = vec![0u64; d];

        for (j, &coeff) in self.coeffs.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            let exp = (k * j) % two_d;
            let idx = exp % d;
            if exp < d {
                result[idx] = add_mod(result[idx], coeff, self.q);
            } else {
                // X^{d+r} ≡ -X^r mod (X^d + 1)
                result[idx] = sub_mod(result[idx], coeff, self.q);
            }
        }

        Self {
            coeffs: result,
            d,
            q: self.q,
        }
    }
}

// ============================================================================
// Sparse Polynomial Multiplication
// ============================================================================

/// Sparse polynomial representation for challenge polynomials
///
/// Stores only nonzero coefficients as (index, sign) pairs where sign ∈ {-1, +1}
#[derive(Clone, Debug)]
pub struct SparseChallenge {
    /// Nonzero entries: (index, sign) where sign is +1 or -1
    pub entries: Vec<(usize, i8)>,
    pub d: usize,
}

impl SparseChallenge {
    /// Create sparse challenge with given nonzero positions and signs
    pub fn new(entries: Vec<(usize, i8)>, d: usize) -> Self {
        Self { entries, d }
    }

    /// Create random sparse challenge with c nonzero ±1 coefficients
    pub fn random<R: rand::Rng>(rng: &mut R, d: usize, c: usize) -> Self {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..d).collect();
        indices.shuffle(rng);
        let entries: Vec<(usize, i8)> = indices
            .iter()
            .take(c)
            .map(|&idx| (idx, if rng.gen_bool(0.5) { 1i8 } else { -1i8 }))
            .collect();
        Self { entries, d }
    }

    /// Convert to dense RingElement
    pub fn to_ring_element(&self, q: u64) -> RingElement {
        let mut coeffs = vec![0i64; self.d];
        for &(idx, sign) in &self.entries {
            coeffs[idx] = sign as i64;
        }
        RingElement::from_signed(coeffs, self.d, q)
    }
}

/// Multiply dense polynomial f by sparse polynomial g
///
/// Time complexity: O(c · d) where c = |g.entries|
///
/// For Hachi challenges with c=16, this is faster than NTT for d ≤ 1024
pub fn sparse_mul(f: &RingElement, g: &SparseChallenge) -> RingElement {
    assert_eq!(f.d, g.d);
    let d = f.d;
    let q = f.q;
    let mut result = vec![0i128; d];

    for &(idx, sign) in &g.entries {
        let s = sign as i128;
        for i in 0..d {
            let j = i + idx;
            if j < d {
                result[j] += s * f.coeffs[i] as i128;
            } else {
                // X^{i+idx} mod (X^d + 1) = -X^{i+idx-d}
                result[j - d] -= s * f.coeffs[i] as i128;
            }
        }
    }

    // Reduce mod q
    let coeffs: Vec<u64> = result
        .iter()
        .map(|&c| {
            let c = c % q as i128;
            ((c + q as i128) % q as i128) as u64
        })
        .collect();

    RingElement { coeffs, d, q }
}

// ============================================================================
// Base-b Decomposition
// ============================================================================

/// Decompose integer a into base-b digits
///
/// Returns δ digits in {0, ..., b-1} such that a = Σ digits[i] · b^i
pub fn decompose_scalar(a: u64, b: u64, delta: usize) -> Vec<u64> {
    let mut digits = Vec::with_capacity(delta);
    let mut rem = a;
    for _ in 0..delta {
        digits.push(rem % b);
        rem /= b;
    }
    digits
}

/// Optimized base-16 decomposition using bit operations
#[inline]
pub fn decompose_base_16(a: u32) -> [u8; 8] {
    [
        (a & 0xF) as u8,
        ((a >> 4) & 0xF) as u8,
        ((a >> 8) & 0xF) as u8,
        ((a >> 12) & 0xF) as u8,
        ((a >> 16) & 0xF) as u8,
        ((a >> 20) & 0xF) as u8,
        ((a >> 24) & 0xF) as u8,
        ((a >> 28) & 0xF) as u8,
    ]
}

/// Decompose polynomial coefficient-wise
///
/// Returns δ polynomials, each with coefficients in {0, ..., b-1}
pub fn decompose_poly(f: &RingElement, b: u64, delta: usize) -> Vec<RingElement> {
    let d = f.d;
    let q = f.q;
    let mut result: Vec<Vec<u64>> = vec![vec![0; d]; delta];

    for i in 0..d {
        let digits = decompose_scalar(f.coeffs[i], b, delta);
        for j in 0..delta {
            result[j][i] = digits[j];
        }
    }

    result
        .into_iter()
        .map(|coeffs| RingElement { coeffs, d, q })
        .collect()
}

/// Recompose polynomial from base-b digits
///
/// Inverse of decompose_poly: f = Σ decomposed[i] · b^i
pub fn recompose_poly(decomposed: &[RingElement], b: u64) -> RingElement {
    assert!(!decomposed.is_empty());
    let d = decomposed[0].d;
    let q = decomposed[0].q;

    let mut result = vec![0u64; d];
    let mut power = 1u64;

    for component in decomposed {
        assert_eq!(component.d, d);
        for i in 0..d {
            result[i] = add_mod(
                result[i],
                mul_mod(component.coeffs[i], power, q),
                q,
            );
        }
        power = mul_mod(power, b, q);
    }

    RingElement { coeffs: result, d, q }
}

// ============================================================================
// Direct NTT (for NTT-friendly primes)
// ============================================================================

/// Precomputed NTT tables for a specific (d, q, ψ)
#[derive(Clone)]
pub struct NttTables {
    pub d: usize,
    pub q: u64,
    /// ψ = primitive 2d-th root of unity mod q
    pub psi: u64,
    /// ψ^{-1} mod q
    pub psi_inv: u64,
    /// d^{-1} mod q
    pub d_inv: u64,
    /// ω = ψ² (primitive d-th root of unity)
    pub omega: u64,
    /// Powers of ψ for forward twist: ψ^0, ψ^1, ..., ψ^{d-1}
    pub psi_powers: Vec<u64>,
    /// Powers of ψ^{-1} for inverse twist
    pub psi_inv_powers: Vec<u64>,
    /// Twiddle factors for Cooley-Tukey: bit-reversed ω powers
    pub twiddles: Vec<u64>,
    /// Inverse twiddle factors for Gentleman-Sande
    pub inv_twiddles: Vec<u64>,
}

impl NttTables {
    /// Create NTT tables for given parameters
    ///
    /// Requires: q ≡ 1 (mod 2d), ψ is a primitive 2d-th root of unity mod q
    pub fn new(d: usize, q: u64, psi: u64) -> Self {
        assert!(d.is_power_of_two());
        // Verify ψ is a primitive 2d-th root of unity:
        // ψ^{2d} = 1 and ψ^d = -1 (not 1)
        debug_assert_eq!(pow_mod(psi, 2 * d as u64, q), 1);
        debug_assert_eq!(pow_mod(psi, d as u64, q), q - 1); // ψ^d = -1 mod q

        let psi_inv = mod_inv(psi, q);
        let d_inv = mod_inv(d as u64, q);
        let omega = mul_mod(psi, psi, q);

        // Precompute ψ powers for twist
        let mut psi_powers = vec![0u64; d];
        let mut psi_inv_powers = vec![0u64; d];
        let mut psi_pow = 1u64;
        let mut psi_inv_pow = 1u64;
        for i in 0..d {
            psi_powers[i] = psi_pow;
            psi_inv_powers[i] = psi_inv_pow;
            psi_pow = mul_mod(psi_pow, psi, q);
            psi_inv_pow = mul_mod(psi_inv_pow, psi_inv, q);
        }

        // Precompute twiddle factors (bit-reversed order)
        let log_d = d.trailing_zeros() as usize;
        let mut twiddles = vec![0u64; d];
        let mut inv_twiddles = vec![0u64; d];
        let omega_inv = mod_inv(omega, q);

        for i in 0..d {
            let rev_i = bit_reverse(i, log_d);
            twiddles[i] = pow_mod(omega, rev_i as u64, q);
            inv_twiddles[i] = pow_mod(omega_inv, rev_i as u64, q);
        }

        Self {
            d,
            q,
            psi,
            psi_inv,
            d_inv,
            omega,
            psi_powers,
            psi_inv_powers,
            twiddles,
            inv_twiddles,
        }
    }

    /// Forward negacyclic NTT: transforms coefficient form to NTT form
    ///
    /// For R_q = Z_q[X]/(X^d + 1), this computes:
    /// â_j = Σ_{i=0}^{d-1} a_i · ψ^{(2j+1)i}  for j = 0,...,d-1
    pub fn forward(&self, a: &mut [u64]) {
        assert_eq!(a.len(), self.d);

        // Pre-twist: multiply a_i by ψ^i
        for i in 0..self.d {
            a[i] = mul_mod(a[i], self.psi_powers[i], self.q);
        }

        // Bit-reverse input for decimation-in-time
        self.bit_reverse_permute(a);

        // Cooley-Tukey NTT with ω = ψ²
        self.cooley_tukey(a);
    }

    /// Inverse negacyclic NTT: transforms NTT form to coefficient form
    ///
    /// Computes: a_i = d^{-1} · Σ_{j=0}^{d-1} â_j · ψ^{-(2j+1)i}
    pub fn inverse(&self, a: &mut [u64]) {
        assert_eq!(a.len(), self.d);

        // Gentleman-Sande INTT (decimation-in-frequency, outputs in bit-reversed order)
        self.gentleman_sande(a);

        // Bit-reverse output
        self.bit_reverse_permute(a);

        // Un-twist: multiply a_i by ψ^{-i} · d^{-1}
        for i in 0..self.d {
            a[i] = mul_mod(a[i], self.psi_inv_powers[i], self.q);
            a[i] = mul_mod(a[i], self.d_inv, self.q);
        }
    }

    /// Bit-reverse permutation in place
    fn bit_reverse_permute(&self, a: &mut [u64]) {
        let n = self.d;
        let log_n = n.trailing_zeros() as usize;
        for i in 0..n {
            let rev_i = bit_reverse(i, log_n);
            if i < rev_i {
                a.swap(i, rev_i);
            }
        }
    }

    /// Cooley-Tukey decimation-in-time NTT
    /// Assumes input is in bit-reversed order, outputs in natural order
    fn cooley_tukey(&self, a: &mut [u64]) {
        let n = self.d;
        let q = self.q;
        let log_n = n.trailing_zeros() as usize;

        for s in 0..log_n {
            let m = 1 << (s + 1); // m = 2, 4, 8, ...
            let half_m = m / 2;
            let w_m = pow_mod(self.omega, (n / m) as u64, q);

            for k in (0..n).step_by(m) {
                let mut w = 1u64;
                for j in 0..half_m {
                    let u = a[k + j];
                    let t = mul_mod(w, a[k + j + half_m], q);
                    a[k + j] = add_mod(u, t, q);
                    a[k + j + half_m] = sub_mod(u, t, q);
                    w = mul_mod(w, w_m, q);
                }
            }
        }
    }

    /// Gentleman-Sande decimation-in-frequency INTT
    /// Assumes input is in natural order, outputs in bit-reversed order
    fn gentleman_sande(&self, a: &mut [u64]) {
        let n = self.d;
        let q = self.q;
        let omega_inv = mod_inv(self.omega, q);
        let log_n = n.trailing_zeros() as usize;

        for s in (0..log_n).rev() {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let w_m = pow_mod(omega_inv, (n / m) as u64, q);

            for k in (0..n).step_by(m) {
                let mut w = 1u64;
                for j in 0..half_m {
                    let u = a[k + j];
                    let v = a[k + j + half_m];
                    a[k + j] = add_mod(u, v, q);
                    a[k + j + half_m] = mul_mod(sub_mod(u, v, q), w, q);
                    w = mul_mod(w, w_m, q);
                }
            }
        }
    }

    /// Pointwise multiply two NTT vectors
    pub fn pointwise_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert_eq!(a.len(), self.d);
        assert_eq!(b.len(), self.d);
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| mul_mod(x, y, self.q))
            .collect()
    }

    /// Pointwise multiply and accumulate: result += a * b
    pub fn pointwise_mul_acc(&self, result: &mut [u64], a: &[u64], b: &[u64]) {
        for i in 0..self.d {
            result[i] = add_mod(result[i], mul_mod(a[i], b[i], self.q), self.q);
        }
    }

    /// Ring multiplication using NTT
    pub fn ring_mul(&self, a: &RingElement, b: &RingElement) -> RingElement {
        assert_eq!(a.d, self.d);
        assert_eq!(b.d, self.d);
        assert_eq!(a.q, self.q);
        assert_eq!(b.q, self.q);

        let mut a_ntt = a.coeffs.clone();
        let mut b_ntt = b.coeffs.clone();

        self.forward(&mut a_ntt);
        self.forward(&mut b_ntt);

        let mut c_ntt = self.pointwise_mul(&a_ntt, &b_ntt);

        self.inverse(&mut c_ntt);

        RingElement {
            coeffs: c_ntt,
            d: self.d,
            q: self.q,
        }
    }
}

/// Bit-reverse an index
fn bit_reverse(x: usize, log_n: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// ============================================================================
// Tower-Optimized Trace (uses automorphisms)
// ============================================================================

/// Compute coset representatives for tower-optimized trace
///
/// For compression factor k (power of 2), returns log_2(k) automorphism indices
/// such that Tr_H = (1 + σ_{g_1})(1 + σ_{g_2})...(1 + σ_{g_t})
pub fn compute_tower_coset_reps(d: usize, compression_k: usize) -> Vec<usize> {
    assert!(d.is_power_of_two());
    assert!(compression_k.is_power_of_two());
    assert!(compression_k <= d);

    let modulus = 2 * d;
    let g: usize = 3; // Generator of (Z/2dZ)^×/{±1}

    let levels = (compression_k as f64).log2() as usize;
    let mut coset_reps = Vec::with_capacity(levels);

    for level in 0..levels {
        // At level i, coset rep is g^{d/(2^{i+1})} mod 2d
        let exp = d / (1 << (level + 1));
        let rep = pow_mod(g as u64, exp as u64, modulus as u64) as usize;
        coset_reps.push(rep);
    }

    coset_reps
}

/// Tower-optimized trace computation
///
/// Computes Tr_H(x) = (1 + σ_{g_1})(1 + σ_{g_2})...(1 + σ_{g_t})(x)
///
/// Time: O(log|H| · d) instead of O(|H| · d) for naive
pub fn trace_tower(x: &RingElement, coset_reps: &[usize]) -> RingElement {
    let mut result = x.clone();

    for &rep in coset_reps {
        let sigma_result = result.automorphism(rep);
        result = result.add(&sigma_result);
    }

    result
}

/// Naive trace computation: Tr_H(x) = Σ_{σ∈H} σ(x)
///
/// Time: O(|H| · d) - used for correctness testing
pub fn trace_naive(x: &RingElement, subgroup_elements: &[usize]) -> RingElement {
    let mut result = RingElement::zero(x.d, x.q);

    for &sigma in subgroup_elements {
        let sigma_x = x.automorphism(sigma);
        result = result.add(&sigma_x);
    }

    result
}

/// Enumerate elements of subgroup H of order k in (Z/2dZ)^×
pub fn enumerate_subgroup(d: usize, k: usize) -> Vec<usize> {
    let modulus = 2 * d;
    let g: usize = 3;

    // H = <g^{d/k}> has order k
    let step = d / k;
    let gen = pow_mod(g as u64, step as u64, modulus as u64) as usize;

    let mut elements = Vec::with_capacity(k);
    let mut current = 1usize;
    for _ in 0..k {
        elements.push(current);
        current = (current * gen) % modulus;
    }

    elements
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(12345)
    }

    #[test]
    fn test_ring_element_add_sub() {
        let mut rng = test_rng();
        let d = 64;
        let q = 3329u64;

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        let sum = a.add(&b);
        let diff = sum.sub(&b);
        assert_eq!(a, diff);

        let zero = RingElement::zero(d, q);
        let neg_a = a.neg();
        assert_eq!(a.add(&neg_a), zero);
    }

    #[test]
    fn test_schoolbook_mul_identity() {
        let d = 32;
        let q = 3329u64;

        // Multiply by 1
        let mut rng = test_rng();
        let a = RingElement::random(&mut rng, d, q);
        let one = RingElement::constant(1, d, q);

        let prod = a.mul_schoolbook(&one);
        assert_eq!(a, prod);
    }

    #[test]
    fn test_schoolbook_mul_x() {
        let d = 8;
        let q = 17u64;

        // a(X) * X should shift coefficients with negation at wrap
        let a = RingElement::new(vec![1, 2, 3, 4, 5, 6, 7, 8], d, q);
        let x = RingElement::new(vec![0, 1, 0, 0, 0, 0, 0, 0], d, q);

        let prod = a.mul_schoolbook(&x);
        // X * (1 + 2X + ... + 8X^7) = X + 2X^2 + ... + 7X^7 + 8X^8
        // X^8 ≡ -1 mod (X^8 + 1), so 8X^8 ≡ -8
        // Result: -8 + X + 2X^2 + ... + 7X^7
        assert_eq!(prod.coeffs[0], q - 8); // -8 mod 17 = 9
        assert_eq!(prod.coeffs[1], 1);
        assert_eq!(prod.coeffs[7], 7);
    }

    #[test]
    fn test_automorphism_identity() {
        let mut rng = test_rng();
        let d = 64;
        let q = 3329u64;

        let a = RingElement::random(&mut rng, d, q);
        let sigma_1 = a.automorphism(1);
        assert_eq!(a, sigma_1);
    }

    #[test]
    fn test_automorphism_neg1() {
        let d = 8;
        let q = 17u64;

        // σ_{-1}: X ↦ X^{-1} = X^{2d-1}
        // For a = X, σ_{-1}(X) = X^{-1 mod 16} = X^{15} ≡ X^{15 mod 8} * sign
        // X^{15} = X^8 · X^7 ≡ -X^7 mod (X^8 + 1)
        let a = RingElement::new(vec![0, 1, 0, 0, 0, 0, 0, 0], d, q);
        let sigma = a.automorphism(2 * d - 1); // -1 mod 2d

        assert_eq!(sigma.coeffs[7], q - 1); // -1 mod q
    }

    #[test]
    fn test_sparse_mul() {
        let d = 64;
        let q = 3329u64;
        let mut rng = test_rng();

        let f = RingElement::random(&mut rng, d, q);
        let g = SparseChallenge::random(&mut rng, d, 16);

        // Compare sparse mul to schoolbook
        let sparse_result = sparse_mul(&f, &g);
        let dense_g = g.to_ring_element(q);
        let schoolbook_result = f.mul_schoolbook(&dense_g);

        assert_eq!(sparse_result, schoolbook_result);
    }

    #[test]
    fn test_decompose_recompose() {
        let d = 32;
        let q = 4294967197u64;
        let b = 16u64;
        let delta = 8;
        let mut rng = test_rng();

        let f = RingElement::random(&mut rng, d, q);
        let decomposed = decompose_poly(&f, b, delta);
        let recomposed = recompose_poly(&decomposed, b);

        assert_eq!(f, recomposed);

        // Check all decomposed coefficients are < b
        for comp in &decomposed {
            for &c in &comp.coeffs {
                assert!(c < b);
            }
        }
    }

    #[test]
    fn test_ntt_roundtrip_dilithium() {
        // Use Dilithium parameters (properly NTT-friendly: q ≡ 1 mod 512)
        let d = 256;
        let q = 8380417u64;
        let psi = 1753u64; // primitive 512th root of unity

        let tables = NttTables::new(d, q, psi);
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, d, q);
        let mut a_ntt = a.coeffs.clone();

        tables.forward(&mut a_ntt);
        tables.inverse(&mut a_ntt);

        assert_eq!(a.coeffs, a_ntt);
    }

    #[test]
    fn test_ntt_roundtrip_falcon() {
        // Use Falcon parameters (properly NTT-friendly: q ≡ 1 mod 1024)
        let d = 512;
        let q = 12289u64;
        let psi = 49u64; // primitive 1024th root of unity

        let tables = NttTables::new(d, q, psi);
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, d, q);
        let mut a_ntt = a.coeffs.clone();

        tables.forward(&mut a_ntt);
        tables.inverse(&mut a_ntt);

        assert_eq!(a.coeffs, a_ntt);
    }

    #[test]
    fn test_ntt_mul_vs_schoolbook() {
        // Use Dilithium parameters (properly NTT-friendly)
        let d = 256;
        let q = 8380417u64;
        let psi = 1753u64;

        let tables = NttTables::new(d, q, psi);
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        let ntt_result = tables.ring_mul(&a, &b);
        let schoolbook_result = a.mul_schoolbook(&b);

        assert_eq!(ntt_result, schoolbook_result);
    }

    #[test]
    fn test_tower_coset_reps() {
        let d = 1024;
        let k = 16;

        let reps = compute_tower_coset_reps(d, k);
        // log_2(16) = 4 levels
        assert_eq!(reps.len(), 4);
    }

    #[test]
    fn test_trace_tower_vs_naive() {
        let d = 64;
        let k = 8;
        let q = 3329u64;
        let mut rng = test_rng();

        let coset_reps = compute_tower_coset_reps(d, k);
        let subgroup = enumerate_subgroup(d, k);

        for _ in 0..5 {
            let x = RingElement::random(&mut rng, d, q);
            let tower_result = trace_tower(&x, &coset_reps);
            let naive_result = trace_naive(&x, &subgroup);
            assert_eq!(tower_result, naive_result);
        }
    }
}

