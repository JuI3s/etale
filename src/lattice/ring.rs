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

use std::fmt;

use itertools::iproduct;

use super::modular::{add_mod, mul_mod, sub_mod};

// ============================================================================
// Cyclotomic Type
// ============================================================================

/// Type of cyclotomic polynomial
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CyclotomicType {
    /// Power-of-2: Φ_{2d}(X) = X^d + 1 where d = 2^α (negacyclic)
    PowerOfTwo { d: usize },
    /// Odd prime: Φ_p(X) = X^{p-1} + X^{p-2} + ... + X + 1
    #[allow(dead_code)]
    OddPrime { p: usize },
    /// Twice odd prime: Φ_{2p}(X) = X^{p-1} - X^{p-2} + ... - X + 1
    #[allow(dead_code)]
    TwiceOddPrime { p: usize },
}

impl CyclotomicType {
    /// Get the degree of the cyclotomic polynomial (= φ(m))
    pub fn degree(&self) -> usize {
        match self {
            Self::PowerOfTwo { d } => *d,
            Self::OddPrime { p } | Self::TwiceOddPrime { p } => p - 1,
        }
    }

    /// Get the conductor m (the cyclotomic index)
    pub fn conductor(&self) -> usize {
        match self {
            Self::PowerOfTwo { d } => 2 * d,
            Self::OddPrime { p } => *p,
            Self::TwiceOddPrime { p } => 2 * p,
        }
    }

    /// Check if this is a power-of-two cyclotomic (negacyclic)
    pub fn is_power_of_two(&self) -> bool {
        matches!(self, Self::PowerOfTwo { .. })
    }

    /// Create a power-of-two cyclotomic type
    pub fn power_of_two(d: usize) -> Self {
        debug_assert!(d.is_power_of_two(), "d must be power of 2");
        Self::PowerOfTwo { d }
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
#[derive(Clone, PartialEq, Eq)]
pub struct RingElement {
    /// Coefficients: c_0 + c_1·X + ... + c_{d-1}·X^{d-1}
    pub coeffs: Vec<u64>,
    /// Cyclotomic type (determines degree and reduction polynomial)
    pub cyclotomic: CyclotomicType,
    /// Modulus q (prime)
    pub q: u64,
}

impl fmt::Debug for RingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingElement {{ {:?}, q: {}, coeffs: [...] }}", self.cyclotomic, self.q)
    }
}

// ============================================================================
// Constructors
// ============================================================================

impl RingElement {
    /// Create new ring element from coefficients
    pub fn new(coeffs: Vec<u64>, cyclotomic: CyclotomicType, q: u64) -> Self {
        let d = cyclotomic.degree();
        let mut c = coeffs;
        c.resize(d, 0);
        c.iter_mut().for_each(|coeff| *coeff %= q);
        Self { coeffs: c, cyclotomic, q }
    }

    /// Create for power-of-two cyclotomic (convenience)
    pub fn new_pow2(coeffs: Vec<u64>, d: usize, q: u64) -> Self {
        Self::new(coeffs, CyclotomicType::power_of_two(d), q)
    }

    /// Create from signed coefficients
    pub fn from_signed(coeffs: Vec<i64>, cyclotomic: CyclotomicType, q: u64) -> Self {
        let unsigned = coeffs.into_iter()
            .map(|c| c.rem_euclid(q as i64) as u64)
            .collect();
        Self::new(unsigned, cyclotomic, q)
    }

    /// Create zero element
    pub fn zero(cyclotomic: CyclotomicType, q: u64) -> Self {
        Self { coeffs: vec![0; cyclotomic.degree()], cyclotomic, q }
    }

    /// Create zero for power-of-two cyclotomic
    pub fn zero_pow2(d: usize, q: u64) -> Self {
        Self::zero(CyclotomicType::power_of_two(d), q)
    }

    /// Create constant element
    pub fn constant(c: u64, cyclotomic: CyclotomicType, q: u64) -> Self {
        let mut coeffs = vec![0; cyclotomic.degree()];
        coeffs[0] = c % q;
        Self { coeffs, cyclotomic, q }
    }

    /// Backward compatibility alias
    #[inline]
    pub fn from_pow2(coeffs: Vec<u64>, d: usize, q: u64) -> Self {
        Self::new_pow2(coeffs, d, q)
    }

    pub fn degree(&self) -> usize { self.cyclotomic.degree() }
    #[inline] pub fn d(&self) -> usize { self.degree() }
}

// ============================================================================
// Random Sampling
// ============================================================================

impl RingElement {
    /// Create random element
    pub fn random<R: rand::Rng>(rng: &mut R, cyclotomic: CyclotomicType, q: u64) -> Self {
        let coeffs = (0..cyclotomic.degree()).map(|_| rng.gen_range(0..q)).collect();
        Self { coeffs, cyclotomic, q }
    }

    /// Create random for power-of-two (convenience)
    pub fn random_pow2<R: rand::Rng>(rng: &mut R, d: usize, q: u64) -> Self {
        Self::random(rng, CyclotomicType::power_of_two(d), q)
    }

    /// Create random element with bounded coefficients
    pub fn random_bounded<R: rand::Rng>(rng: &mut R, cyclotomic: CyclotomicType, q: u64, bound: u64) -> Self {
        let coeffs = (0..cyclotomic.degree()).map(|_| rng.gen_range(0..bound)).collect();
        Self::new(coeffs, cyclotomic, q)
    }

    /// Create random ternary element with `num_nonzero` coefficients in {-1, +1}
    pub fn random_ternary<R: rand::Rng>(rng: &mut R, cyclotomic: CyclotomicType, q: u64, num_nonzero: usize) -> Self {
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
// Centered Representation & Norms
// ============================================================================

impl RingElement {
    /// Get centered representation (coefficients in [-q/2, q/2))
    pub fn centered(&self) -> Vec<i64> {
        let half_q = self.q as i64 / 2;
        self.coeffs.iter()
            .map(|&c| {
                let c = c as i64;
                if c > half_q { c - self.q as i64 } else { c }
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
        debug_assert_eq!(self.cyclotomic, other.cyclotomic);
        debug_assert_eq!(self.q, other.q);
        let coeffs = self.coeffs.iter().zip(&other.coeffs)
            .map(|(&a, &b)| add_mod(a, b, self.q))
            .collect();
        Self { coeffs, cyclotomic: self.cyclotomic, q: self.q }
    }

    /// Subtract two ring elements
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.cyclotomic, other.cyclotomic);
        debug_assert_eq!(self.q, other.q);
        let coeffs = self.coeffs.iter().zip(&other.coeffs)
            .map(|(&a, &b)| sub_mod(a, b, self.q))
            .collect();
        Self { coeffs, cyclotomic: self.cyclotomic, q: self.q }
    }

    /// Negate ring element
    pub fn neg(&self) -> Self {
        let coeffs = self.coeffs.iter()
            .map(|&c| if c == 0 { 0 } else { self.q - c })
            .collect();
        Self { coeffs, cyclotomic: self.cyclotomic, q: self.q }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let coeffs = self.coeffs.iter()
            .map(|&c| mul_mod(c, scalar, self.q))
            .collect();
        Self { coeffs, cyclotomic: self.cyclotomic, q: self.q }
    }
}

// ============================================================================
// Polynomial Multiplication
// ============================================================================

impl RingElement {
    /// Polynomial multiplication with reduction mod the cyclotomic polynomial
    pub fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.cyclotomic, other.cyclotomic);
        debug_assert_eq!(self.q, other.q);

        match self.cyclotomic {
            CyclotomicType::PowerOfTwo { d } => self.mul_negacyclic(other, d),
            CyclotomicType::OddPrime { p } => self.mul_odd_prime(other, p),
            CyclotomicType::TwiceOddPrime { p } => self.mul_twice_odd_prime(other, p),
        }
    }

    /// Negacyclic multiplication for X^d + 1
    fn mul_negacyclic(&self, other: &Self, d: usize) -> Self {
        let q = self.q;
        let mut result = vec![0u64; d];

        // X^d ≡ -1, so X^k for k ≥ d maps to -X^{k-d}
        let self_nz = self.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);
        let other_nz = || other.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);

        for ((i, &ai), (j, &bj)) in iproduct!(self_nz, other_nz()) {
            let prod = mul_mod(ai, bj, q);
            let (idx, sign) = if i + j < d { (i + j, 1) } else { (i + j - d, -1) };

            result[idx] = if sign > 0 {
                add_mod(result[idx], prod, q)
            } else {
                sub_mod(result[idx], prod, q)
            };
        }

        Self { coeffs: result, cyclotomic: self.cyclotomic, q }
    }

    /// Multiplication for Φ_p(X) = X^{p-1} + ... + 1
    fn mul_odd_prime(&self, other: &Self, p: usize) -> Self {
        let d = p - 1;
        let q = self.q;
        let mut unreduced = vec![0u64; 2 * d - 1];

        // Schoolbook multiplication (unreduced)
        let self_nz = self.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);
        let other_nz = || other.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);

        for ((i, &ai), (j, &bj)) in iproduct!(self_nz, other_nz()) {
            unreduced[i + j] = add_mod(unreduced[i + j], mul_mod(ai, bj, q), q);
        }

        // Reduce: X^{p-1} ≡ -(1 + X + ... + X^{p-2})
        let mut result = unreduced[..d].to_vec();
        for (k, &coeff) in unreduced.iter().enumerate().skip(d).filter(|(_, &c)| c != 0) {
            for i in 0..d {
                result[(k - d + i) % d] = sub_mod(result[(k - d + i) % d], coeff, q);
            }
        }

        Self { coeffs: result, cyclotomic: self.cyclotomic, q }
    }

    /// Multiplication for Φ_{2p}(X) = X^{p-1} - X^{p-2} + ... + 1
    fn mul_twice_odd_prime(&self, other: &Self, p: usize) -> Self {
        let d = p - 1;
        let q = self.q;
        let mut unreduced = vec![0u64; 2 * d - 1];

        // Schoolbook multiplication (unreduced)
        let self_nz = self.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);
        let other_nz = || other.coeffs.iter().enumerate().filter(|(_, &c)| c != 0);

        for ((i, &ai), (j, &bj)) in iproduct!(self_nz, other_nz()) {
            unreduced[i + j] = add_mod(unreduced[i + j], mul_mod(ai, bj, q), q);
        }

        // Reduce with alternating signs: X^{p-1} ≡ X^{p-2} - X^{p-3} + ... - 1
        let mut result = unreduced[..d].to_vec();
        for (k, &coeff) in unreduced.iter().enumerate().skip(d).filter(|(_, &c)| c != 0) {
            for i in 0..d {
                let target = k - d + i;
                if target < d {
                    result[target] = if i % 2 == 0 {
                        add_mod(result[target], coeff, q)
                    } else {
                        sub_mod(result[target], coeff, q)
                    };
                }
            }
        }

        Self { coeffs: result, cyclotomic: self.cyclotomic, q }
    }
}

// ============================================================================
// Automorphisms
// ============================================================================

impl RingElement {
    /// Apply automorphism σ_k: X ↦ X^k
    pub fn automorphism(&self, k: usize) -> Self {
        let d = self.degree();
        let q = self.q;

        match self.cyclotomic {
            CyclotomicType::PowerOfTwo { .. } => {
                let mut result = vec![0u64; d];
                for (j, &coeff) in self.coeffs.iter().enumerate() {
                    if coeff == 0 { continue; }
                    let exp = (k * j) % (2 * d);
                    let idx = exp % d;
                    if exp < d {
                        result[idx] = add_mod(result[idx], coeff, q);
                    } else {
                        result[idx] = sub_mod(result[idx], coeff, q);
                    }
                }
                Self { coeffs: result, cyclotomic: self.cyclotomic, q }
            }
            CyclotomicType::OddPrime { p } | CyclotomicType::TwiceOddPrime { p } => {
                let m = self.cyclotomic.conductor();
                let mut result = Self::zero(self.cyclotomic, q);
                for (j, &coeff) in self.coeffs.iter().enumerate() {
                    if coeff == 0 { continue; }
                    let new_exp = (k * j) % m;
                    let mut term = Self::zero(self.cyclotomic, q);
                    term.coeffs[new_exp % d] = coeff;
                    result = result.add(&term);
                }
                let _ = p;
                result
            }
        }
    }
}

// ============================================================================
// Sparse Ternary Polynomials
// ============================================================================

/// Sparse ternary polynomial with coefficients in {-1, 0, +1}
///
/// Used for challenge polynomials in lattice-based protocols where
/// the challenge has few nonzero entries, each being ±1.
#[derive(Clone, Debug)]
pub struct SparseTernary {
    /// Nonzero entries: (index, sign) where sign is +1 or -1
    pub entries: Vec<(usize, i8)>,
    /// Polynomial degree
    pub d: usize,
}

impl SparseTernary {
    /// Create sparse ternary polynomial with given entries
    pub fn new(entries: Vec<(usize, i8)>, d: usize) -> Self {
        Self { entries, d }
    }

    /// Create random sparse ternary with c nonzero ±1 coefficients
    pub fn random<R: rand::Rng>(rng: &mut R, d: usize, c: usize) -> Self {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..d).collect();
        indices.shuffle(rng);
        let entries = indices.iter()
            .take(c)
            .map(|&idx| (idx, if rng.gen_bool(0.5) { 1i8 } else { -1i8 }))
            .collect();
        Self { entries, d }
    }

    /// Convert to dense RingElement (power-of-two cyclotomic)
    pub fn to_ring_element(&self, q: u64) -> RingElement {
        let mut coeffs = vec![0i64; self.d];
        for &(idx, sign) in &self.entries {
            coeffs[idx] = sign as i64;
        }
        RingElement::from_signed(coeffs, CyclotomicType::power_of_two(self.d), q)
    }

    /// Multiply dense polynomial f by this sparse ternary (negacyclic)
    ///
    /// Time complexity: O(c · d) where c = number of nonzero entries
    pub fn mul(&self, f: &RingElement) -> RingElement {
        debug_assert_eq!(f.degree(), self.d);
        let (d, q) = (self.d, f.q);
        let mut result = vec![0i128; d];

        for &(idx, sign) in &self.entries {
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

        let coeffs = result.iter()
            .map(|&c| c.rem_euclid(q as i128) as u64)
            .collect();

        RingElement { coeffs, cyclotomic: f.cyclotomic, q }
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

        assert_eq!(a, a.add(&b).sub(&b));
        assert_eq!(RingElement::zero(cyc, q), a.add(&a.neg()));
    }

    #[test]
    fn test_pow2_mul_identity() {
        let cyc = CyclotomicType::power_of_two(32);
        let q = 17u64;

        let mut rng = test_rng();
        let a = RingElement::random(&mut rng, cyc, q);
        let one = RingElement::constant(1, cyc, q);

        assert_eq!(a, a.mul(&one));
    }

    #[test]
    fn test_pow2_mul_x() {
        let cyc = CyclotomicType::power_of_two(8);
        let q = 17u64;

        let a = RingElement::new((1..=8).collect(), cyc, q);
        let mut x_coeffs = vec![0u64; 8];
        x_coeffs[1] = 1;
        let x = RingElement::new(x_coeffs, cyc, q);

        let prod = a.mul(&x);
        // X^8 ≡ -1, so result is -8 + X + 2X^2 + ... + 7X^7
        assert_eq!(prod.coeffs[0], q - 8);
        assert_eq!(prod.coeffs[1], 1);
        assert_eq!(prod.coeffs[7], 7);
    }

    #[test]
    fn test_pow2_automorphism() {
        let cyc = CyclotomicType::power_of_two(64);
        let q = 3329u64;
        let mut rng = test_rng();

        let a = RingElement::random(&mut rng, cyc, q);
        assert_eq!(a, a.automorphism(1)); // σ_1 is identity
    }

    #[test]
    fn test_backward_compat() {
        let (q, d) = (17u64, 8);
        let a = RingElement::from_pow2(vec![1, 2, 3], d, q);
        let b = RingElement::new_pow2(vec![1, 2, 3], d, q);
        assert_eq!(a, b);
        assert_eq!(a.d(), d);
    }

    #[test]
    fn test_centered() {
        let cyc = CyclotomicType::power_of_two(4);
        let q = 17u64;

        // Small positive stays positive
        let a = RingElement::new(vec![1, 2, 3, 4], cyc, q);
        assert_eq!(a.centered(), vec![1, 2, 3, 4]);

        // Values > q/2 become negative
        let b = RingElement::new(vec![16, 15, 9, 8], cyc, q);
        assert_eq!(b.centered(), vec![-1, -2, -8, 8]);

        // Zero stays zero
        assert_eq!(RingElement::zero(cyc, q).centered(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_linf_norm() {
        let cyc = CyclotomicType::power_of_two(4);
        let q = 17u64;

        let a = RingElement::new(vec![1, 2, 3, 4], cyc, q);
        assert_eq!(a.linf_norm(), 4);

        // 16 → -1, 15 → -2
        let b = RingElement::new(vec![16, 15, 1, 0], cyc, q);
        assert_eq!(b.linf_norm(), 2);

        // 9 → -8
        let c = RingElement::new(vec![0, 0, 9, 0], cyc, q);
        assert_eq!(c.linf_norm(), 8);

        assert_eq!(RingElement::zero(cyc, q).linf_norm(), 0);
    }

    #[test]
    fn test_from_signed_rem_euclid() {
        let cyc = CyclotomicType::power_of_two(4);
        let q = 17u64;

        // Positive values within range
        let a = RingElement::from_signed(vec![1, 2, 3, 4], cyc, q);
        assert_eq!(a.coeffs, vec![1, 2, 3, 4]);

        // Negative values: rem_euclid always returns non-negative
        let b = RingElement::from_signed(vec![-1, -5, -17, -18], cyc, q);
        assert_eq!(b.coeffs, vec![16, 12, 0, 16]);

        // Values larger than q get reduced
        let c = RingElement::from_signed(vec![17, 18, 34, -34], cyc, q);
        assert_eq!(c.coeffs, vec![0, 1, 0, 0]);
    }
}
