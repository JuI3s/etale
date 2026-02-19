//! NTT-based ring arithmetic for power-of-two cyclotomics
//!
//! This module provides Number Theoretic Transform (NTT) operations for
//! negacyclic rings R_q = Z_q[X]/(X^d + 1) where **d must be a power of 2**.
//!
//! # Contents
//!
//! - [`RingElement`]: Polynomial in R_q with NTT-accelerated or schoolbook multiplication
//! - [`NegacyclicNtt`]: Precomputed tables for fast negacyclic NTT (power-of-2 only)
//! - [`SparseTernary`]: Sparse ternary polynomials for challenge multiplication
//!
//! # Mathematical Background
//!
//! For R_q = Z_q[X]/(X^d + 1) where d = 2^n:
//! - X^d + 1 = Φ_{2d} is the 2d-th cyclotomic polynomial
//! - Roots are primitive 2d-th roots of unity: ω^{2j+1} for j = 0,...,d-1
//! - **NTT-friendly:** q ≡ 1 (mod 2d) enables direct negacyclic NTT
//! - **Non-NTT-friendly:** Falls back to schoolbook multiplication
//!
//! See [`NegacyclicNtt`] for the power-of-2 limitation details.
//!
//! # References
//!
//! - [Hachi](https://eprint.iacr.org/2026/156) - DQZZ26
//! - Greyhound (CRYPTO 2024) - NS24
//! - NIST FIPS 203/204/206

use std::fmt;

use itertools::iproduct;

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
#[derive(Clone, PartialEq, Eq)]
pub struct RingElement {
    /// Coefficients: c_0 + c_1·X + ... + c_{d-1}·X^{d-1}
    pub coeffs: Vec<u64>,
    /// Ring dimension d (must be power of 2)
    pub d: usize,
    /// Modulus q
    pub q: u64,
}

impl fmt::Debug for RingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RingElement {{ d: {}, q: {}, coeffs: [...] }}",
            self.d, self.q
        )
    }
}

// ============================================================================
// Constructors
// ============================================================================

impl RingElement {
    /// Create new ring element from coefficients
    pub fn new(coeffs: Vec<u64>, d: usize, q: u64) -> Self {
        debug_assert!(d.is_power_of_two(), "d must be power of two");
        let mut c = coeffs;
        c.resize(d, 0);
        for coeff in &mut c {
            *coeff %= q;
        }
        Self { coeffs: c, d, q }
    }

    /// Create from signed coefficients
    pub fn from_signed(coeffs: Vec<i64>, d: usize, q: u64) -> Self {
        let unsigned = coeffs
            .into_iter()
            .map(|c| c.rem_euclid(q as i64) as u64)
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

    /// Create constant element
    pub fn constant(c: u64, d: usize, q: u64) -> Self {
        let mut coeffs = vec![0; d];
        coeffs[0] = c % q;
        Self { coeffs, d, q }
    }

    /// Create random element
    pub fn random<R: rand::Rng>(rng: &mut R, d: usize, q: u64) -> Self {
        Self {
            coeffs: (0..d).map(|_| rng.gen_range(0..q)).collect(),
            d,
            q,
        }
    }

    /// Create random element with bounded coefficients
    pub fn random_bounded<R: rand::Rng>(rng: &mut R, d: usize, q: u64, bound: u64) -> Self {
        Self::new((0..d).map(|_| rng.gen_range(0..bound)).collect(), d, q)
    }

    /// Create random ternary element with `num_nonzero` coefficients in {-1, +1}
    pub fn random_ternary<R: rand::Rng>(rng: &mut R, d: usize, q: u64, num_nonzero: usize) -> Self {
        use rand::seq::SliceRandom;
        let mut coeffs = vec![0i64; d];
        let mut indices: Vec<usize> = (0..d).collect();
        indices.shuffle(rng);
        for &idx in indices.iter().take(num_nonzero) {
            coeffs[idx] = if rng.gen_bool(0.5) { 1 } else { -1 };
        }
        Self::from_signed(coeffs, d, q)
    }
}

// ============================================================================
// Centered Representation & Norms
// ============================================================================

impl RingElement {
    /// Get centered representation (coefficients in [-q/2, q/2))
    pub fn centered(&self) -> Vec<i64> {
        let half_q = self.q as i64 / 2;
        self.coeffs
            .iter()
            .map(|&c| {
                let c = c as i64;
                if c > half_q {
                    c - self.q as i64
                } else {
                    c
                }
            })
            .collect()
    }

    /// Compute ℓ∞ norm of centered representation
    pub fn linf_norm(&self) -> u64 {
        self.centered()
            .iter()
            .map(|c| c.unsigned_abs())
            .max()
            .expect("ring element must have positive dimension")
    }
}

// ============================================================================
// Basic Arithmetic
// ============================================================================

impl RingElement {
    /// Add two ring elements
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.d, other.d);
        debug_assert_eq!(self.q, other.q);
        let coeffs = self
            .coeffs
            .iter()
            .zip(&other.coeffs)
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
        debug_assert_eq!(self.d, other.d);
        debug_assert_eq!(self.q, other.q);
        let coeffs = self
            .coeffs
            .iter()
            .zip(&other.coeffs)
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
        let coeffs = self
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
        let coeffs = self
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
    pub fn mul_schoolbook(&self, other: &Self) -> Self {
        debug_assert_eq!(self.d, other.d);
        debug_assert_eq!(self.q, other.q);
        let (d, q) = (self.d, self.q);
        let mut result = vec![0u64; d];

        // Negacyclic convolution: X^d ≡ -1, so X^k for k ≥ d maps to -X^{k-d}
        let self_nz = self.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);
        let other_nz = || other.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);

        for ((i, &ai), (j, &bj)) in iproduct!(self_nz, other_nz()) {
            let prod = mul_mod(ai, bj, q);
            let (idx, sign) = if i + j < d {
                (i + j, 1)
            } else {
                (i + j - d, -1)
            };

            result[idx] = if sign > 0 {
                add_mod(result[idx], prod, q)
            } else {
                sub_mod(result[idx], prod, q)
            };
        }

        Self {
            coeffs: result,
            d,
            q,
        }
    }
}

// ============================================================================
// Sparse Polynomial (re-exported from ring module)
// ============================================================================

pub use super::ring::SparseTernary;

/// Multiply dense polynomial f by sparse ternary polynomial g
///
/// Time complexity: O(c · d) where c = |g.entries|
pub fn sparse_mul(f: &RingElement, g: &SparseTernary) -> RingElement {
    debug_assert_eq!(f.d, g.d);
    let (d, q) = (f.d, f.q);
    let mut result = vec![0i128; d];

    for &(idx, sign) in &g.entries {
        let s = sign as i128;
        for (i, &coeff) in f.coeffs.iter().enumerate() {
            let j = i + idx;
            if j < d {
                result[j] += s * coeff as i128;
            } else {
                result[j - d] -= s * coeff as i128;
            }
        }
    }

    let coeffs = result
        .iter()
        .map(|&c| c.rem_euclid(q as i128) as u64)
        .collect();

    RingElement { coeffs, d, q }
}

// ============================================================================
// Negacyclic NTT (Power-of-2 Only)
// ============================================================================

/// Precomputed tables for negacyclic NTT over R_q = Z_q[X]/(X^d + 1).
///
/// # Power-of-2 Limitation
///
/// **This implementation requires d = 2^n (power-of-two).**
///
/// The negacyclic structure X^d + 1 = Φ_{2d} only holds when d is a power of 2.
/// Specifically, this implementation assumes:
/// - Binary-split butterfly: d halves evenly at every level
/// - Bit-reverse permutation: uses exactly log₂(d) bits
/// - NTT-friendliness: q ≡ 1 (mod 2d)
/// - Negacyclic ring: X^d ≡ -1, giving the sign flip in butterflies
///
/// For general cyclotomic rings R_q = Z_q[X]/Φ_m with m not a power of 2,
/// use [`super::ring::RingElement`] which supports schoolbook multiplication
/// for arbitrary cyclotomics (Φ_p, Φ_{2p}, etc.).
///
/// # See Also
///
/// - [`super::trace`]: Galois tower trace works for arbitrary cyclotomics
/// - [`super::ring`]: General cyclotomic rings without NTT restriction
#[derive(Clone, Debug)]
pub struct NegacyclicNtt {
    pub d: usize,
    pub q: u64,
    /// ψ = primitive 2d-th root of unity mod q
    pub psi: u64,
    pub psi_inv: u64,
    pub d_inv: u64,
    /// ω = ψ² (primitive d-th root)
    pub omega: u64,
    /// Powers of ψ for forward twist
    pub psi_powers: Vec<u64>,
    /// Powers of ψ^{-1} for inverse twist
    pub psi_inv_powers: Vec<u64>,
    /// Twiddle factors (bit-reversed ω powers)
    pub twiddles: Vec<u64>,
    pub inv_twiddles: Vec<u64>,
}

impl NegacyclicNtt {
    /// Create NTT tables for given parameters
    ///
    /// Requires: q ≡ 1 (mod 2d), ψ is a primitive 2d-th root of unity
    pub fn new(d: usize, q: u64, psi: u64) -> Self {
        // Hard assert: non-power-of-2 silently corrupts results (see module docs)
        assert!(
            d.is_power_of_two(),
            "NTT requires d to be a power of 2, got d={d}"
        );
        debug_assert_eq!(pow_mod(psi, 2 * d as u64, q), 1);
        debug_assert_eq!(pow_mod(psi, d as u64, q), q - 1);

        let psi_inv = mod_inv(psi, q);
        let d_inv = mod_inv(d as u64, q);
        let omega = mul_mod(psi, psi, q);
        let omega_inv = mod_inv(omega, q);
        let log_d = d.trailing_zeros() as usize;

        // Precompute ψ powers
        let psi_powers = std::iter::successors(Some(1u64), |&p| Some(mul_mod(p, psi, q)))
            .take(d)
            .collect();
        let psi_inv_powers = std::iter::successors(Some(1u64), |&p| Some(mul_mod(p, psi_inv, q)))
            .take(d)
            .collect();

        // Precompute twiddle factors (bit-reversed order)
        let twiddles = (0..d)
            .map(|i| pow_mod(omega, bit_reverse(i, log_d) as u64, q))
            .collect();
        let inv_twiddles = (0..d)
            .map(|i| pow_mod(omega_inv, bit_reverse(i, log_d) as u64, q))
            .collect();

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

    /// Forward negacyclic NTT
    pub fn forward(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.d);

        // Pre-twist: multiply a_i by ψ^i
        for (i, (coeff, &psi_pow)) in a.iter_mut().zip(&self.psi_powers).enumerate() {
            *coeff = mul_mod(*coeff, psi_pow, self.q);
            let _ = i;
        }

        self.bit_reverse_permute(a);
        self.cooley_tukey(a);
    }

    /// Inverse negacyclic NTT
    pub fn inverse(&self, a: &mut [u64]) {
        debug_assert_eq!(a.len(), self.d);

        self.gentleman_sande(a);
        self.bit_reverse_permute(a);

        // Un-twist: multiply a_i by ψ^{-i} · d^{-1}
        for (coeff, &psi_inv_pow) in a.iter_mut().zip(&self.psi_inv_powers) {
            *coeff = mul_mod(*coeff, psi_inv_pow, self.q);
            *coeff = mul_mod(*coeff, self.d_inv, self.q);
        }
    }

    fn bit_reverse_permute(&self, a: &mut [u64]) {
        let log_n = self.d.trailing_zeros() as usize;
        for i in 0..self.d {
            let rev_i = bit_reverse(i, log_n);
            if i < rev_i {
                a.swap(i, rev_i);
            }
        }
    }

    /// Cooley-Tukey decimation-in-time NTT
    fn cooley_tukey(&self, a: &mut [u64]) {
        let (n, q) = (self.d, self.q);
        let log_n = n.trailing_zeros() as usize;

        for s in 0..log_n {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let w_m = pow_mod(self.omega, (n / m) as u64, q);

            for chunk in a.chunks_mut(m) {
                let (lo, hi) = chunk.split_at_mut(half_m);
                let mut w = 1u64;

                for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                    // Cooley-Tukey butterfly: (u, v, ω^j) → (u + ω^j·v, u - ω^j·v)
                    //
                    // Given half-size NTT evaluations u_j = a_even(η^j) and v_j = a_odd(η^j)
                    // where η = ω² is the primitive (d/2)-th root, the two outputs are
                    // evaluations of a(X) at the conjugate pair ω^j and ω^{j+d/2}.
                    //
                    // The sign flip in the second output comes from ω^{d/2} = -1
                    // (the unique involution in μ_d, since ω is a primitive d-th root).
                    let t = mul_mod(w, *v, q);
                    (*u, *v) = (add_mod(*u, t, q), sub_mod(*u, t, q));
                    w = mul_mod(w, w_m, q);
                }
            }
        }
    }

    /// Gentleman-Sande decimation-in-frequency INTT
    fn gentleman_sande(&self, a: &mut [u64]) {
        let (n, q) = (self.d, self.q);
        let omega_inv = mod_inv(self.omega, q);
        let log_n = n.trailing_zeros() as usize;

        for s in (0..log_n).rev() {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let w_m = pow_mod(omega_inv, (n / m) as u64, q);

            for chunk in a.chunks_mut(m) {
                let (lo, hi) = chunk.split_at_mut(half_m);
                let mut w = 1u64;

                for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                    // Gentleman-Sande butterfly: (u, v) → (u + v, (u - v)·ω^{-j})
                    //
                    // This is the inverse of the Cooley-Tukey butterfly: add/subtract
                    // first, then multiply — reversed from CT's multiply then add/subtract.
                    //
                    // The twiddle factor uses ω^{-1} to invert the forward transform.
                    let sum = add_mod(*u, *v, q);
                    let diff = mul_mod(sub_mod(*u, *v, q), w, q);
                    (*u, *v) = (sum, diff);
                    w = mul_mod(w, w_m, q);
                }
            }
        }
    }

    /// Pointwise multiply two NTT vectors
    pub fn pointwise_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        debug_assert_eq!(a.len(), self.d);
        debug_assert_eq!(b.len(), self.d);
        a.iter()
            .zip(b)
            .map(|(&x, &y)| mul_mod(x, y, self.q))
            .collect()
    }

    /// Pointwise multiply and accumulate: result += a * b
    pub fn pointwise_mul_acc(&self, result: &mut [u64], a: &[u64], b: &[u64]) {
        for ((r, &ai), &bi) in result.iter_mut().zip(a).zip(b) {
            *r = add_mod(*r, mul_mod(ai, bi, self.q), self.q);
        }
    }

    /// Ring multiplication using NTT
    pub fn ring_mul(&self, a: &RingElement, b: &RingElement) -> RingElement {
        debug_assert_eq!(a.d, self.d);
        debug_assert_eq!(b.d, self.d);
        debug_assert_eq!(a.q, self.q);
        debug_assert_eq!(b.q, self.q);

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
    (0..log_n).fold(0, |acc, i| (acc << 1) | ((x >> i) & 1))
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

    /// Standard NTT parameters: (q, d, ψ, name)
    const NTT_PARAMS: &[(u64, usize, u64, &str)] = &[
        (8_380_417, 256, 1753, "Dilithium"),
        (12289, 512, 49, "Falcon"),
    ];

    fn assert_primitive_root(q: u64, d: usize, psi: u64, name: &str) {
        assert_eq!(
            pow_mod(psi, 2 * d as u64, q),
            1,
            "{name}: ψ^(2d) should be 1"
        );
        assert_eq!(pow_mod(psi, d as u64, q), q - 1, "{name}: ψ^d should be -1");
        assert_eq!(q % (2 * d as u64), 1, "{name}: q should be ≡ 1 (mod 2d)");
    }

    #[test]
    fn test_primitive_roots() {
        for &(q, d, psi, name) in NTT_PARAMS {
            assert_primitive_root(q, d, psi, name);
        }
    }

    #[test]
    fn test_ring_element_add_sub() {
        let mut rng = test_rng();
        let (d, q) = (64, 3329u64);

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        assert_eq!(a, a.add(&b).sub(&b));
        assert_eq!(RingElement::zero(d, q), a.add(&a.neg()));
    }

    #[test]
    fn test_schoolbook_mul_identity() {
        let (d, q) = (32, 3329u64);
        let mut rng = test_rng();
        let a = RingElement::random(&mut rng, d, q);
        let one = RingElement::constant(1, d, q);
        assert_eq!(a, a.mul_schoolbook(&one));
    }

    #[test]
    fn test_schoolbook_mul_x() {
        let (d, q) = (8, 17u64);
        let a = RingElement::new(vec![1, 2, 3, 4, 5, 6, 7, 8], d, q);
        let x = RingElement::new(vec![0, 1, 0, 0, 0, 0, 0, 0], d, q);

        let prod = a.mul_schoolbook(&x);
        // X^8 ≡ -1, so result is -8 + X + 2X^2 + ... + 7X^7
        assert_eq!(prod.coeffs[0], q - 8);
        assert_eq!(prod.coeffs[1], 1);
        assert_eq!(prod.coeffs[7], 7);
    }

    #[test]
    fn test_sparse_mul() {
        let (d, q) = (64, 3329u64);
        let mut rng = test_rng();

        let f = RingElement::random(&mut rng, d, q);
        let g = SparseTernary::random(&mut rng, d, 16);

        // Convert sparse to dense ntt::RingElement for comparison
        let g_dense = {
            let mut coeffs = vec![0i64; d];
            for &(idx, sign) in &g.entries {
                coeffs[idx] = sign as i64;
            }
            RingElement::from_signed(coeffs, d, q)
        };

        assert_eq!(sparse_mul(&f, &g), f.mul_schoolbook(&g_dense));
    }

    #[test]
    fn test_ntt_roundtrip_dilithium() {
        let (d, q, psi) = (256, 8_380_417_u64, 1753u64);
        let tables = NegacyclicNtt::new(d, q, psi);
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, d, q);
        let mut a_ntt = a.coeffs.clone();

        tables.forward(&mut a_ntt);
        tables.inverse(&mut a_ntt);

        assert_eq!(a.coeffs, a_ntt);
    }

    #[test]
    fn test_ntt_roundtrip_falcon() {
        let (d, q, psi) = (512, 12289u64, 49u64);
        let tables = NegacyclicNtt::new(d, q, psi);
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, d, q);
        let mut a_ntt = a.coeffs.clone();

        tables.forward(&mut a_ntt);
        tables.inverse(&mut a_ntt);

        assert_eq!(a.coeffs, a_ntt);
    }

    #[test]
    fn test_ntt_mul_vs_schoolbook() {
        let (d, q, psi) = (256, 8_380_417_u64, 1753u64);
        let tables = NegacyclicNtt::new(d, q, psi);
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        assert_eq!(tables.ring_mul(&a, &b), a.mul_schoolbook(&b));
    }

    #[test]
    fn test_centered() {
        let (d, q) = (4, 17u64);

        // Small positive stays positive
        let a = RingElement::new(vec![1, 2, 3, 4], d, q);
        assert_eq!(a.centered(), vec![1, 2, 3, 4]);

        // Values > q/2 become negative
        let b = RingElement::new(vec![16, 15, 9, 8], d, q);
        // 16 > 8 → 16-17 = -1, 15 > 8 → -2, 9 > 8 → -8, 8 = 8 → 8
        assert_eq!(b.centered(), vec![-1, -2, -8, 8]);

        // Zero stays zero
        let c = RingElement::zero(d, q);
        assert_eq!(c.centered(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_linf_norm() {
        let (d, q) = (4, 17u64);

        // Small values
        let a = RingElement::new(vec![1, 2, 3, 4], d, q);
        assert_eq!(a.linf_norm(), 4);

        // Values that wrap to negative (16 mod 17 → -1, 15 → -2)
        let b = RingElement::new(vec![16, 15, 1, 0], d, q);
        assert_eq!(b.linf_norm(), 2); // max(|-1|, |-2|, |1|, |0|) = 2

        // Larger centered negative
        let c = RingElement::new(vec![0, 0, 9, 0], d, q);
        // 9 > 8 → 9-17 = -8
        assert_eq!(c.linf_norm(), 8);

        // Zero element
        assert_eq!(RingElement::zero(d, q).linf_norm(), 0);
    }

    #[test]
    fn test_from_signed_rem_euclid() {
        let (d, q) = (4, 17u64);

        // Positive values within range
        let a = RingElement::from_signed(vec![1, 2, 3, 4], d, q);
        assert_eq!(a.coeffs, vec![1, 2, 3, 4]);

        // Negative values: rem_euclid always returns non-negative
        // -1.rem_euclid(17) = 16, -5.rem_euclid(17) = 12
        let b = RingElement::from_signed(vec![-1, -5, -17, -18], d, q);
        assert_eq!(b.coeffs, vec![16, 12, 0, 16]);

        // Values larger than q get reduced
        let c = RingElement::from_signed(vec![17, 18, 34, -34], d, q);
        assert_eq!(c.coeffs, vec![0, 1, 0, 0]);

        // Edge cases: verify rem_euclid vs % operator behavior
        // Rust's % can return negative: -22 % 17 = -5
        // But rem_euclid always returns [0, q): -22.rem_euclid(17) = 12
        assert_eq!((-22i64).rem_euclid(17), 12);
        assert_eq!((-22i64) % 17, -5); // contrast with % operator
    }
}
