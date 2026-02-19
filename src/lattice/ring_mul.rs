//! Ring multiplication backends for R_q = Z_q[X]/(X^d + 1)
//!
//! This module provides a trait-based abstraction over ring multiplication,
//! allowing the same commitment code to work with different backends:
//!
//! - [`SchoolbookBackend`]: O(d²) — always available, no constraints on q
//! - [`NttBackend`]: O(d log d) — requires NTT-friendly q ≡ 1 (mod 2d)
//!
//! # Future Extensions
//!
//! `TfheNttBackend` (CRT-based multiplication for non-NTT-friendly q like Hachi's)
//! will be added when tfhe-ntt is integrated. The trait is designed for this:
//! tfhe-ntt uses auxiliary NTT-friendly primes and CRT reconstruction to achieve
//! O(d log d) multiplication even when q ≢ 1 (mod 2d).

use super::ntt::{NegacyclicNtt, RingElement};

// ============================================================================
// Ring Multiplication Trait
// ============================================================================

/// Backend for ring multiplication in R_q = Z_q[X]/(X^d + 1).
///
/// Implementations provide different trade-offs:
/// - Schoolbook: O(d²), always works
/// - NTT: O(d log d), requires q ≡ 1 (mod 2d)
/// - TfheNtt (future): O(d log d) via CRT, works for any q
pub trait RingMulBackend {
    /// Multiply two ring elements in R_q.
    fn ring_mul(&self, a: &RingElement, b: &RingElement) -> RingElement;
}

// ============================================================================
// Schoolbook Backend
// ============================================================================

/// Schoolbook O(d²) multiplication — always available.
///
/// No constraints on the modulus q. Works for Hachi, Greyhound, and all
/// parameter sets regardless of NTT-friendliness.
#[derive(Clone, Copy, Debug, Default)]
pub struct SchoolbookBackend;

impl RingMulBackend for SchoolbookBackend {
    fn ring_mul(&self, a: &RingElement, b: &RingElement) -> RingElement {
        a.mul_schoolbook(b)
    }
}

// ============================================================================
// NTT Backend
// ============================================================================

/// NTT-based O(d log d) multiplication — requires NTT-friendly q.
///
/// The modulus must satisfy q ≡ 1 (mod 2d) for direct negacyclic NTT.
/// Use with Dilithium, Falcon, Kyber, or other NTT-friendly parameter sets.
#[derive(Clone, Debug)]
pub struct NttBackend {
    /// Precomputed NTT tables for (d, q, ψ)
    pub tables: NegacyclicNtt,
}

impl NttBackend {
    /// Create NTT backend from precomputed tables.
    pub fn new(tables: NegacyclicNtt) -> Self {
        Self { tables }
    }
}

impl RingMulBackend for NttBackend {
    fn ring_mul(&self, a: &RingElement, b: &RingElement) -> RingElement {
        self.tables.ring_mul(a, b)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::params::DILITHIUM_2;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn schoolbook_backend() {
        let mut rng = test_rng();
        let (d, q) = (64, 65537);

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        let backend = SchoolbookBackend;
        let result = backend.ring_mul(&a, &b);

        // Verify against direct call
        assert_eq!(result, a.mul_schoolbook(&b));
    }

    #[test]
    fn ntt_backend() {
        let mut rng = test_rng();

        // Dilithium is NTT-friendly: q = 8380417 ≡ 1 (mod 512)
        let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        let tables = NegacyclicNtt::new(d, q, psi);
        let backend = NttBackend::new(tables);
        let result = backend.ring_mul(&a, &b);

        // Verify against schoolbook
        assert_eq!(result, a.mul_schoolbook(&b));
    }

    #[test]
    fn ntt_vs_schoolbook_agree() {
        let mut rng = test_rng();

        // Dilithium params (NTT-friendly)
        let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);

        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        let schoolbook = SchoolbookBackend;
        let ntt = NttBackend::new(NegacyclicNtt::new(d, q, psi));

        let result_schoolbook = schoolbook.ring_mul(&a, &b);
        let result_ntt = ntt.ring_mul(&a, &b);

        assert_eq!(result_schoolbook, result_ntt);
    }
}
