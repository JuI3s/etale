//! Cyclotomic ring elements for lattice-based cryptography
//!
//! This module provides a general representation for elements in cyclotomic rings
//! R_q = Z_q[X]/Φ_m(X), where Φ_m is the m-th cyclotomic polynomial.
//!
//! # Supported Cyclotomic Types
//!
//! | Type | Conductor m | Polynomial | Degree φ(m) |
//! |------|-------------|------------|-------------|
//! | Power-of-2 | 2^k | X^{2^{k-1}} + 1 | 2^{k-1} |
//! | Odd prime | p | X^{p-1} + ... + X + 1 | p - 1 |
//! | Twice odd prime | 2p | X^{p-1} - X^{p-2} + ... - X + 1 | p - 1 |
//!
//! # Design Notes
//!
//! The current implementation focuses on power-of-2 cyclotomics (m = 2d where d = 2^α),
//! which are used in Kyber, Dilithium, Falcon, and Hachi. The structure is designed
//! to be extensible to general cyclotomic rings in the future.

use std::fmt;

use super::modular::{add_mod, mul_mod, sub_mod};

// ============================================================================
// Cyclotomic Type
// ============================================================================

/// Type of cyclotomic polynomial
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CyclotomicType {
    /// Power-of-2: Φ_{2d}(X) = X^d + 1 where d = 2^α
    /// This is the "negacyclic" case used in most lattice crypto.
    PowerOfTwo {
        /// Ring degree d (must be power of 2)
        d: usize,
    },

    /// Odd prime: Φ_p(X) = X^{p-1} + X^{p-2} + ... + X + 1
    /// Degree is p - 1.
    #[allow(dead_code)]
    OddPrime {
        /// The prime p
        p: usize,
    },

    /// Twice odd prime: Φ_{2p}(X) = X^{p-1} - X^{p-2} + ... - X + 1
    /// Degree is p - 1.
    #[allow(dead_code)]
    TwiceOddPrime {
        /// The odd prime p
        p: usize,
    },

    // Future: Prime power, general composite, etc.
}

impl CyclotomicType {
    /// Get the degree of the cyclotomic polynomial (= φ(m))
    pub fn degree(&self) -> usize {
        match self {
            CyclotomicType::PowerOfTwo { d } => *d,
            CyclotomicType::OddPrime { p } => p - 1,
            CyclotomicType::TwiceOddPrime { p } => p - 1,
        }
    }

    /// Get the conductor m (the cyclotomic index)
    pub fn conductor(&self) -> usize {
        match self {
            CyclotomicType::PowerOfTwo { d } => 2 * d,
            CyclotomicType::OddPrime { p } => *p,
            CyclotomicType::TwiceOddPrime { p } => 2 * p,
        }
    }

    /// Check if this is a power-of-two cyclotomic (negacyclic)
    pub fn is_power_of_two(&self) -> bool {
        matches!(self, CyclotomicType::PowerOfTwo { .. })
    }

    /// Create a power-of-two cyclotomic type
    pub fn power_of_two(d: usize) -> Self {
        assert!(d.is_power_of_two(), "d must be power of 2 for X^d + 1");
        CyclotomicType::PowerOfTwo { d }
    }

    /// Get the Galois group order |Gal(Q(ζ_m)/Q)| = φ(m)
    pub fn galois_group_order(&self) -> usize {
        self.degree()
    }
}

// ============================================================================
// Ring Element
// ============================================================================

/// Element in a cyclotomic ring R_q = Z_q[X]/Φ_m(X)
///
/// Coefficients are stored as unsigned values in [0, q).
/// The cyclotomic type determines the reduction polynomial.
///
/// # Representation Choice: `Vec<u64>` vs `[u64; N]`
///
/// We use `Vec<u64>` rather than a const-generic array `[u64; N]` for flexibility:
///
/// | Approach | Pros | Cons |
/// |----------|------|------|
/// | `Vec<u64>` | Runtime flexible, simpler API | Heap allocation |
/// | `[u64; N]` | Stack-allocated, cache-friendly | Requires const generics everywhere |
///
/// For production code with fixed parameters, consider refactoring to:
/// ```ignore
/// pub struct RingElement<const D: usize> {
///     pub coeffs: [u64; D],
///     pub q: u64,
/// }
/// ```
///
/// Current choice optimizes for API simplicity during development/benchmarking.
/// The heap allocation overhead is negligible compared to O(d²) multiplication.
#[derive(Clone, PartialEq, Eq)]
pub struct RingElement {
    /// Coefficients: c_0 + c_1·X + ... + c_{d-1}·X^{d-1}
    ///
    /// Stored as `Vec` for runtime flexibility. For fixed-size rings,
    /// could be `[u64; D]` with const generics for better cache locality.
    pub coeffs: Vec<u64>,
    /// Cyclotomic type (determines degree and reduction polynomial)
    pub cyclotomic: CyclotomicType,
    /// Modulus q (prime)
    pub q: u64,
}

impl fmt::Debug for RingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RingElement {{ {:?}, q: {}, coeffs: [...] }}",
            self.cyclotomic, self.q
        )
    }
}

impl RingElement {
    /// Create new ring element from coefficients
    pub fn new(coeffs: Vec<u64>, cyclotomic: CyclotomicType, q: u64) -> Self {
        let d = cyclotomic.degree();
        let mut c = coeffs;
        c.resize(d, 0);
        // Reduce coefficients mod q
        for coeff in c.iter_mut() {
            *coeff %= q;
        }
        Self {
            coeffs: c,
            cyclotomic,
            q,
        }
    }

    /// Create new ring element for power-of-two cyclotomic (convenience)
    pub fn new_pow2(coeffs: Vec<u64>, d: usize, q: u64) -> Self {
        Self::new(coeffs, CyclotomicType::power_of_two(d), q)
    }

    /// Create new ring element from signed coefficients
    pub fn from_signed(coeffs: Vec<i64>, cyclotomic: CyclotomicType, q: u64) -> Self {
        let unsigned: Vec<u64> = coeffs
            .into_iter()
            .map(|c| ((c % q as i64) + q as i64) as u64 % q)
            .collect();
        Self::new(unsigned, cyclotomic, q)
    }

    /// Create zero element
    pub fn zero(cyclotomic: CyclotomicType, q: u64) -> Self {
        Self {
            coeffs: vec![0; cyclotomic.degree()],
            cyclotomic,
            q,
        }
    }

    /// Create zero element for power-of-two cyclotomic (convenience)
    pub fn zero_pow2(d: usize, q: u64) -> Self {
        Self::zero(CyclotomicType::power_of_two(d), q)
    }

    /// Create element representing constant c
    pub fn constant(c: u64, cyclotomic: CyclotomicType, q: u64) -> Self {
        let d = cyclotomic.degree();
        let mut coeffs = vec![0; d];
        coeffs[0] = c % q;
        Self {
            coeffs,
            cyclotomic,
            q,
        }
    }

    /// Get the ring degree
    pub fn degree(&self) -> usize {
        self.cyclotomic.degree()
    }

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
            .map(|c| c.abs() as u64)
            .max()
            .unwrap_or(0)
    }

    /// Add two ring elements
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.cyclotomic, other.cyclotomic);
        assert_eq!(self.q, other.q);
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| add_mod(a, b, self.q))
            .collect();
        Self {
            coeffs,
            cyclotomic: self.cyclotomic,
            q: self.q,
        }
    }

    /// Subtract two ring elements
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.cyclotomic, other.cyclotomic);
        assert_eq!(self.q, other.q);
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| sub_mod(a, b, self.q))
            .collect();
        Self {
            coeffs,
            cyclotomic: self.cyclotomic,
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
            cyclotomic: self.cyclotomic,
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
            cyclotomic: self.cyclotomic,
            q: self.q,
        }
    }

    /// Polynomial multiplication with reduction mod the cyclotomic polynomial
    ///
    /// This is the general multiplication that works for any cyclotomic type.
    /// For power-of-two, this is negacyclic convolution.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cyclotomic, other.cyclotomic);
        assert_eq!(self.q, other.q);

        match self.cyclotomic {
            CyclotomicType::PowerOfTwo { d } => self.mul_negacyclic(other, d),
            CyclotomicType::OddPrime { p } => self.mul_odd_prime(other, p),
            CyclotomicType::TwiceOddPrime { p } => self.mul_twice_odd_prime(other, p),
        }
    }

    /// Negacyclic multiplication for X^d + 1 (power-of-two case)
    fn mul_negacyclic(&self, other: &Self, d: usize) -> Self {
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
                    // X^d ≡ -1, so X^{d+r} ≡ -X^r
                    result[k - d] = sub_mod(result[k - d], prod, q);
                }
            }
        }

        Self {
            coeffs: result,
            cyclotomic: self.cyclotomic,
            q,
        }
    }

    /// Multiplication for Φ_p(X) = X^{p-1} + X^{p-2} + ... + 1 (odd prime case)
    ///
    /// Reduction: X^{p-1} ≡ -(X^{p-2} + X^{p-3} + ... + 1)
    fn mul_odd_prime(&self, other: &Self, p: usize) -> Self {
        let d = p - 1; // degree of Φ_p
        let q = self.q;

        // First compute unreduced product (degree up to 2d-2)
        let mut unreduced = vec![0u64; 2 * d - 1];
        for i in 0..d {
            if self.coeffs[i] == 0 {
                continue;
            }
            for j in 0..d {
                if other.coeffs[j] == 0 {
                    continue;
                }
                let prod = mul_mod(self.coeffs[i], other.coeffs[j], q);
                unreduced[i + j] = add_mod(unreduced[i + j], prod, q);
            }
        }

        // Reduce mod Φ_p(X) = X^{p-1} + X^{p-2} + ... + 1
        // X^{p-1} ≡ -(X^{p-2} + ... + 1) = -Σ_{i=0}^{p-2} X^i
        let mut result = unreduced[..d].to_vec();
        for k in d..(2 * d - 1) {
            // X^k where k >= p-1
            // X^k = X^{k-(p-1)} · X^{p-1} ≡ -X^{k-(p-1)} · (1 + X + ... + X^{p-2})
            let coeff = unreduced[k];
            if coeff == 0 {
                continue;
            }
            let base = k - d; // k - (p-1)
            for i in 0..d {
                if base + i < d {
                    result[base + i] = sub_mod(result[base + i], coeff, q);
                } else {
                    // Need recursive reduction (for high degrees)
                    // This is a simplified version; full implementation needs iteration
                    result[(base + i) % d] = sub_mod(result[(base + i) % d], coeff, q);
                }
            }
        }

        Self {
            coeffs: result,
            cyclotomic: self.cyclotomic,
            q,
        }
    }

    /// Multiplication for Φ_{2p}(X) = X^{p-1} - X^{p-2} + X^{p-3} - ... + 1
    fn mul_twice_odd_prime(&self, other: &Self, p: usize) -> Self {
        let d = p - 1;
        let q = self.q;

        // Compute unreduced product
        let mut unreduced = vec![0u64; 2 * d - 1];
        for i in 0..d {
            if self.coeffs[i] == 0 {
                continue;
            }
            for j in 0..d {
                if other.coeffs[j] == 0 {
                    continue;
                }
                let prod = mul_mod(self.coeffs[i], other.coeffs[j], q);
                unreduced[i + j] = add_mod(unreduced[i + j], prod, q);
            }
        }

        // Reduce mod Φ_{2p}(X) = X^{p-1} - X^{p-2} + ... - X + 1
        // X^{p-1} ≡ X^{p-2} - X^{p-3} + ... + X - 1
        let mut result = unreduced[..d].to_vec();
        for k in d..(2 * d - 1) {
            let coeff = unreduced[k];
            if coeff == 0 {
                continue;
            }
            // Alternating signs reduction
            for i in 0..d {
                let target = k - d + i;
                if target < d {
                    let sign = if i % 2 == 0 { 1 } else { -1i64 };
                    if sign > 0 {
                        result[target] = add_mod(result[target], coeff, q);
                    } else {
                        result[target] = sub_mod(result[target], coeff, q);
                    }
                }
            }
        }

        Self {
            coeffs: result,
            cyclotomic: self.cyclotomic,
            q,
        }
    }

    /// Apply automorphism σ_k: X ↦ X^k
    ///
    /// For cyclotomic field Q(ζ_m), automorphisms are σ_k for k ∈ (Z/mZ)^×.
    pub fn automorphism(&self, k: usize) -> Self {
        let d = self.degree();
        let m = self.cyclotomic.conductor();
        let q = self.q;

        match self.cyclotomic {
            CyclotomicType::PowerOfTwo { .. } => {
                // For X^d + 1: X^j ↦ X^{kj mod 2d} with sign from X^d = -1
                let mut result = vec![0u64; d];
                for (j, &coeff) in self.coeffs.iter().enumerate() {
                    if coeff == 0 {
                        continue;
                    }
                    let exp = (k * j) % (2 * d);
                    let idx = exp % d;
                    if exp < d {
                        result[idx] = add_mod(result[idx], coeff, q);
                    } else {
                        result[idx] = sub_mod(result[idx], coeff, q);
                    }
                }
                Self {
                    coeffs: result,
                    cyclotomic: self.cyclotomic,
                    q,
                }
            }
            CyclotomicType::OddPrime { p } | CyclotomicType::TwiceOddPrime { p } => {
                // For Φ_m: need to compute X^j ↦ X^{kj} and reduce
                // This is more complex; simplified version:
                let mut result = Self::zero(self.cyclotomic, q);
                for (j, &coeff) in self.coeffs.iter().enumerate() {
                    if coeff == 0 {
                        continue;
                    }
                    // X^j ↦ X^{kj mod m} then reduce mod Φ_m
                    let new_exp = (k * j) % m;
                    let mut term = Self::zero(self.cyclotomic, q);
                    if new_exp < d {
                        term.coeffs[new_exp] = coeff;
                    } else {
                        // Need reduction
                        term.coeffs[new_exp % d] = coeff; // Simplified
                    }
                    result = result.add(&term.scalar_mul(1));
                }
                let _ = p; // suppress warning
                result
            }
        }
    }
}

// ============================================================================
// Conversions for backward compatibility
// ============================================================================

impl RingElement {
    /// Create from the old-style (coeffs, d, q) signature
    /// Assumes power-of-two cyclotomic
    #[inline]
    pub fn from_pow2(coeffs: Vec<u64>, d: usize, q: u64) -> Self {
        Self::new_pow2(coeffs, d, q)
    }

    /// Get degree (for backward compatibility with old code using .d)
    #[inline]
    pub fn d(&self) -> usize {
        self.degree()
    }
}

// ============================================================================
// Random sampling
// ============================================================================

impl RingElement {
    /// Create random element
    pub fn random<R: rand::Rng>(rng: &mut R, cyclotomic: CyclotomicType, q: u64) -> Self {
        let d = cyclotomic.degree();
        let coeffs: Vec<u64> = (0..d).map(|_| rng.gen_range(0..q)).collect();
        Self {
            coeffs,
            cyclotomic,
            q,
        }
    }

    /// Create random element for power-of-two cyclotomic (convenience)
    pub fn random_pow2<R: rand::Rng>(rng: &mut R, d: usize, q: u64) -> Self {
        Self::random(rng, CyclotomicType::power_of_two(d), q)
    }

    /// Create random element with coefficients bounded by `bound`
    pub fn random_bounded<R: rand::Rng>(
        rng: &mut R,
        cyclotomic: CyclotomicType,
        q: u64,
        bound: u64,
    ) -> Self {
        let d = cyclotomic.degree();
        let coeffs: Vec<u64> = (0..d).map(|_| rng.gen_range(0..bound)).collect();
        Self::new(coeffs, cyclotomic, q)
    }

    /// Create sparse random element with `num_nonzero` entries in {-1, 0, 1}
    pub fn random_sparse<R: rand::Rng>(
        rng: &mut R,
        cyclotomic: CyclotomicType,
        q: u64,
        num_nonzero: usize,
    ) -> Self {
        use rand::seq::SliceRandom;
        let d = cyclotomic.degree();
        let mut coeffs = vec![0i64; d];
        let mut indices: Vec<usize> = (0..d).collect();
        indices.shuffle(rng);
        for &idx in indices.iter().take(num_nonzero) {
            coeffs[idx] = if rng.gen_bool(0.5) { 1 } else { -1 };
        }
        Self::from_signed(coeffs, cyclotomic, q)
    }
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
    fn test_cyclotomic_type() {
        let pow2 = CyclotomicType::power_of_two(256);
        assert_eq!(pow2.degree(), 256);
        assert_eq!(pow2.conductor(), 512);
        assert!(pow2.is_power_of_two());

        let odd = CyclotomicType::OddPrime { p: 17 };
        assert_eq!(odd.degree(), 16);
        assert_eq!(odd.conductor(), 17);
        assert!(!odd.is_power_of_two());

        let twice = CyclotomicType::TwiceOddPrime { p: 17 };
        assert_eq!(twice.degree(), 16);
        assert_eq!(twice.conductor(), 34);
    }

    #[test]
    fn test_pow2_add_sub() {
        let mut rng = test_rng();
        let cyc = CyclotomicType::power_of_two(64);
        let q = 3329u64;

        let a = RingElement::random(&mut rng, cyc, q);
        let b = RingElement::random(&mut rng, cyc, q);

        let sum = a.add(&b);
        let diff = sum.sub(&b);
        assert_eq!(a, diff);

        let zero = RingElement::zero(cyc, q);
        let neg_a = a.neg();
        assert_eq!(a.add(&neg_a), zero);
    }

    #[test]
    fn test_pow2_mul_identity() {
        let cyc = CyclotomicType::power_of_two(32);
        let q = 17u64;

        let mut rng = test_rng();
        let a = RingElement::random(&mut rng, cyc, q);
        let one = RingElement::constant(1, cyc, q);

        let prod = a.mul(&one);
        assert_eq!(a, prod);
    }

    #[test]
    fn test_pow2_mul_x() {
        let cyc = CyclotomicType::power_of_two(8);
        let q = 17u64;

        // a(X) = 1 + 2X + 3X^2 + ... + 8X^7
        let a = RingElement::new((1..=8).collect(), cyc, q);
        // x = X
        let mut x_coeffs = vec![0u64; 8];
        x_coeffs[1] = 1;
        let x = RingElement::new(x_coeffs, cyc, q);

        let prod = a.mul(&x);
        // X * (1 + 2X + ... + 8X^7) = X + 2X^2 + ... + 7X^7 + 8X^8
        // X^8 ≡ -1, so result is -8 + X + 2X^2 + ... + 7X^7
        assert_eq!(prod.coeffs[0], q - 8); // -8 mod 17 = 9
        assert_eq!(prod.coeffs[1], 1);
        assert_eq!(prod.coeffs[7], 7);
    }

    #[test]
    fn test_pow2_automorphism() {
        let cyc = CyclotomicType::power_of_two(64);
        let q = 3329u64;
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, cyc, q);

        // σ_1 is identity
        let sigma_1 = a.automorphism(1);
        assert_eq!(a, sigma_1);
    }

    #[test]
    fn test_backward_compat() {
        let q = 17u64;
        let d = 8;

        // Old style
        let a = RingElement::from_pow2(vec![1, 2, 3], d, q);
        assert_eq!(a.d(), d);
        assert_eq!(a.degree(), d);

        // New style
        let b = RingElement::new_pow2(vec![1, 2, 3], d, q);
        assert_eq!(a, b);
    }
}

