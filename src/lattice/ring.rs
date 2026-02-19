//! Ring trait and NTT-domain ring element
//!
//! Two concrete types implement [`Ring`]:
//! - [`RingElement`](crate::lattice::ntt::RingElement) (coefficient domain, O(d²) mul)
//! - [`NttRingElement`] (NTT/CRT domain, O(d) mul)
//!
//! The type system prevents mixing representations.
//! Use [`NttContext`] to convert between them.

use std::fmt::Debug;

use rand::Rng;

use crate::lattice::ntt::{NegacyclicNtt, RingElement};

// ============================================================================
// Ring Trait
// ============================================================================

/// Trait for elements of R_q = Z_q[X]/(X^d + 1)
///
/// Implementations may store elements in coefficient or NTT form.
/// The type system enforces that operations only combine elements
/// in the same representation.
///
/// Within each representation, the multiplication algorithm is fixed:
/// - RingElement: schoolbook O(d²)
/// - NttRingElement: pointwise O(d)
///
/// There is no runtime strategy dispatch. The algorithm is determined
/// by the type.
pub trait Ring: Clone + PartialEq + Eq + Debug + Sized {
    /// Ring dimension d
    fn degree(&self) -> usize;

    /// Modulus q
    fn modulus(&self) -> u64;

    /// Addition in R_q
    fn add(&self, other: &Self) -> Self;

    /// Subtraction in R_q
    fn sub(&self, other: &Self) -> Self;

    /// Multiplication in R_q
    fn mul(&self, other: &Self) -> Self;

    /// Negation in R_q
    fn neg(&self) -> Self;

    /// Additive identity
    fn zero(d: usize, q: u64) -> Self;

    /// Sample uniformly random element
    fn random<R: Rng>(rng: &mut R, d: usize, q: u64) -> Self;

    /// Sample with bounded coefficients (for short vectors)
    /// For NttRingElement this will panic — bounded sampling
    /// requires coefficient domain.
    fn random_bounded<R: Rng>(rng: &mut R, d: usize, q: u64, bound: u64) -> Self;
}

// ============================================================================
// NttRingElement
// ============================================================================

/// Ring element in NTT/CRT representation.
///
/// Stores evaluations (a(ω⁰), a(ω¹), ..., a(ω^{d-1})).
/// Multiplication is O(d) pointwise. Requires NTT-friendly q.
///
/// Must be created from RingElement via [`NttContext`]. Cannot be
/// constructed directly from coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NttRingElement {
    /// Evaluations at roots of unity
    pub evals: Vec<u64>,
    /// Ring dimension d
    pub d: usize,
    /// Modulus q
    pub q: u64,
}

impl Ring for NttRingElement {
    fn degree(&self) -> usize {
        self.d
    }

    fn modulus(&self) -> u64 {
        self.q
    }

    fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.d, other.d);
        debug_assert_eq!(self.q, other.q);
        let evals = self
            .evals
            .iter()
            .zip(&other.evals)
            .map(|(&a, &b)| {
                let sum = a as u128 + b as u128;
                (sum % self.q as u128) as u64
            })
            .collect();
        NttRingElement {
            evals,
            d: self.d,
            q: self.q,
        }
    }

    fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.d, other.d);
        debug_assert_eq!(self.q, other.q);
        let evals = self
            .evals
            .iter()
            .zip(&other.evals)
            .map(|(&a, &b)| if a >= b { a - b } else { self.q - b + a })
            .collect();
        NttRingElement {
            evals,
            d: self.d,
            q: self.q,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.d, other.d);
        debug_assert_eq!(self.q, other.q);
        let evals = self
            .evals
            .iter()
            .zip(&other.evals)
            .map(|(&a, &b)| ((a as u128 * b as u128) % self.q as u128) as u64)
            .collect();
        NttRingElement {
            evals,
            d: self.d,
            q: self.q,
        }
    }

    fn neg(&self) -> Self {
        let evals = self
            .evals
            .iter()
            .map(|&a| if a == 0 { 0 } else { self.q - a })
            .collect();
        NttRingElement {
            evals,
            d: self.d,
            q: self.q,
        }
    }

    fn zero(d: usize, q: u64) -> Self {
        NttRingElement {
            evals: vec![0; d],
            d,
            q,
        }
    }

    fn random<R: Rng>(rng: &mut R, d: usize, q: u64) -> Self {
        // Random in NTT domain = random evaluations
        // This is a uniformly random ring element (just in NTT form)
        let evals = (0..d).map(|_| rng.gen_range(0..q)).collect();
        NttRingElement { evals, d, q }
    }

    fn random_bounded<R: Rng>(_rng: &mut R, _d: usize, _q: u64, _bound: u64) -> Self {
        // Bounded sampling requires coefficient domain — bound on coefficients
        // has no meaning in NTT domain. Use RingElement::random_bounded then
        // convert via NttContext::forward.
        panic!(
            "random_bounded requires coefficient domain. \
             Use RingElement::random_bounded then NttContext::forward."
        )
    }
}

// ============================================================================
// NttContext
// ============================================================================

/// Converts between coefficient and NTT domain.
///
/// Wraps NegacyclicNtt. This is the only way to create NttRingElement
/// from RingElement and vice versa.
///
/// # Example
///
/// ```ignore
/// let ctx = NttContext::new(256, 8380417, 1753);
/// let coeff = RingElement::random(&mut rng, 256, 8380417);
/// let ntt = ctx.forward(&coeff);        // O(d log d)
/// let back = ctx.inverse(&ntt);         // O(d log d)
/// assert_eq!(coeff, back);
/// ```
#[derive(Clone, Debug)]
pub struct NttContext {
    tables: NegacyclicNtt,
    /// Ring dimension d
    pub d: usize,
    /// Modulus q
    pub q: u64,
}

impl NttContext {
    /// Create NTT context for R_q = Z_q[X]/(X^d + 1).
    ///
    /// Requires NTT-friendly parameters: q ≡ 1 (mod 2d),
    /// psi is a primitive 2d-th root of unity mod q.
    pub fn new(d: usize, q: u64, psi: u64) -> Self {
        Self {
            tables: NegacyclicNtt::new(d, q, psi),
            d,
            q,
        }
    }

    /// Coefficient → NTT domain: O(d log d)
    pub fn forward(&self, elem: &RingElement) -> NttRingElement {
        debug_assert_eq!(elem.d, self.d);
        debug_assert_eq!(elem.q, self.q);
        let mut evals = elem.coeffs.clone();
        self.tables.forward(&mut evals);
        NttRingElement {
            evals,
            d: self.d,
            q: self.q,
        }
    }

    /// NTT → Coefficient domain: O(d log d)
    pub fn inverse(&self, elem: &NttRingElement) -> RingElement {
        debug_assert_eq!(elem.d, self.d);
        debug_assert_eq!(elem.q, self.q);
        let mut coeffs = elem.evals.clone();
        self.tables.inverse(&mut coeffs);
        RingElement {
            coeffs,
            d: self.d,
            q: self.q,
        }
    }

    /// Forward-transform an entire matrix (for converting AjtaiKey to NTT domain)
    pub fn forward_matrix(&self, matrix: &[Vec<RingElement>]) -> Vec<Vec<NttRingElement>> {
        matrix
            .iter()
            .map(|row| row.iter().map(|e| self.forward(e)).collect())
            .collect()
    }

    /// Forward-transform a vector
    pub fn forward_vec(&self, v: &[RingElement]) -> Vec<NttRingElement> {
        v.iter().map(|e| self.forward(e)).collect()
    }

    /// Inverse-transform a vector
    pub fn inverse_vec(&self, v: &[NttRingElement]) -> Vec<RingElement> {
        v.iter().map(|e| self.inverse(e)).collect()
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
        use rand::seq::SliceRandom as _;
        let mut indices: Vec<usize> = (0..d).collect();
        indices.shuffle(rng);
        let entries = indices
            .iter()
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
        RingElement::from_signed(coeffs, self.d, q)
    }

    /// Multiply dense polynomial f by this sparse ternary (negacyclic)
    ///
    /// Time complexity: O(c · d) where c = number of nonzero entries
    pub fn mul(&self, f: &RingElement) -> RingElement {
        debug_assert_eq!(f.d, self.d);
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

        let coeffs = result
            .iter()
            .map(|&c| c.rem_euclid(q as i128) as u64)
            .collect();

        RingElement { coeffs, d, q }
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
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn ntt_roundtrip() {
        let mut rng = test_rng();
        let (d, q, psi) = (256, 8_380_417, 1753); // Dilithium params
        let ctx = NttContext::new(d, q, psi);

        let a = RingElement::random(&mut rng, d, q);
        let ntt = ctx.forward(&a);
        let back = ctx.inverse(&ntt);

        assert_eq!(a, back);
    }

    #[test]
    fn ntt_mul_matches_schoolbook() {
        let mut rng = test_rng();
        let (d, q, psi) = (256, 8_380_417, 1753);
        let ctx = NttContext::new(d, q, psi);

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        // Schoolbook in coefficient domain
        let c_school = a.mul_schoolbook(&b);

        // Pointwise in NTT domain, then inverse
        let a_ntt = ctx.forward(&a);
        let b_ntt = ctx.forward(&b);
        let c_ntt = a_ntt.mul(&b_ntt);
        let c_back = ctx.inverse(&c_ntt);

        assert_eq!(c_school, c_back);
    }

    #[test]
    fn ntt_add_matches_coefficient() {
        let mut rng = test_rng();
        let (d, q, psi) = (256, 8_380_417, 1753);
        let ctx = NttContext::new(d, q, psi);

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        let sum_coeff = a.add(&b);
        let sum_ntt = ctx.forward(&a).add(&ctx.forward(&b));
        let sum_back = ctx.inverse(&sum_ntt);

        assert_eq!(sum_coeff, sum_back);
    }

    #[test]
    fn ntt_zero() {
        let (d, q) = (256, 8_380_417);
        let z = NttRingElement::zero(d, q);
        assert!(z.evals.iter().all(|&e| e == 0));
    }

    #[test]
    #[should_panic(expected = "random_bounded requires coefficient domain")]
    fn ntt_random_bounded_panics() {
        let mut rng = test_rng();
        NttRingElement::random_bounded(&mut rng, 256, 8_380_417, 16);
    }

    #[test]
    fn ntt_sub_matches_coefficient() {
        let mut rng = test_rng();
        let (d, q, psi) = (256, 8_380_417, 1753);
        let ctx = NttContext::new(d, q, psi);

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        let diff_coeff = a.sub(&b);
        let diff_ntt = ctx.forward(&a).sub(&ctx.forward(&b));
        let diff_back = ctx.inverse(&diff_ntt);

        assert_eq!(diff_coeff, diff_back);
    }
}
