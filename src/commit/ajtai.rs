//! Ajtai/SIS-based commitment scheme
//!
//! # Core Interface
//!
//! [`AjtaiKey<R>`] is a random matrix A ∈ R_q^{κ × m}, generic over ring
//! representation. The commitment to a short vector s is t = A · s.
//!
//! - `AjtaiKey<RingElement>`: coefficient domain, O(d²) mul
//! - `AjtaiKey<NttRingElement>`: NTT domain, O(d) mul
//!
//! Use [`AjtaiKey::to_ntt`] to convert a coefficient-domain key to NTT domain.
//!
//! # Decomposition Workflow
//!
//! [`commit_decomposed`] and [`verify_decomposed`] handle the base-b
//! decomposition workflow. These are coefficient-domain only (type-enforced).
//!
//! # Security
//!
//! Binding relies on Module-SIS hardness: finding short s₁ ≠ s₂ with A·s₁ = A·s₂.
//!
//! # References
//!
//! - [Hachi](https://eprint.iacr.org/2026/156) - DQZZ26

use crate::lattice::decompose::decompose_poly;
use crate::lattice::ntt::RingElement;
use crate::lattice::ring::{NttContext, NttRingElement, Ring};

// ============================================================================
// Ajtai Commitment Key
// ============================================================================

/// Ajtai commitment key: matrix A ∈ R_q^{κ × m}
///
/// Generic over ring representation:
/// - `AjtaiKey<RingElement>`: coefficient domain, schoolbook mul O(d²)
/// - `AjtaiKey<NttRingElement>`: NTT domain, pointwise mul O(d)
#[derive(Clone, Debug)]
pub struct AjtaiKey<R: Ring> {
    /// Number of rows κ (module rank)
    pub rows: usize,
    /// Number of columns m
    pub cols: usize,
    /// Ring dimension d
    pub d: usize,
    /// Modulus q
    pub q: u64,
    /// Matrix A: rows × cols array of ring elements
    pub matrix: Vec<Vec<R>>,
}

impl<R: Ring> AjtaiKey<R> {
    /// Generate random commitment key.
    ///
    /// Creates a random κ × m matrix over R_q = Z_q[X]/(X^d + 1).
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `rows` - Number of rows κ (module rank)
    /// * `cols` - Number of columns m
    /// * `d` - Ring dimension
    /// * `q` - Modulus
    pub fn random<Rng: rand::Rng>(
        rng: &mut Rng,
        rows: usize,
        cols: usize,
        d: usize,
        q: u64,
    ) -> Self {
        let matrix = (0..rows)
            .map(|_| (0..cols).map(|_| R::random(rng, d, q)).collect())
            .collect();

        Self {
            rows,
            cols,
            d,
            q,
            matrix,
        }
    }

    /// Commit to a short vector: t = A · s
    ///
    /// # Arguments
    ///
    /// * `s` - Short vector of ring elements with length `self.cols`
    ///
    /// # Returns
    ///
    /// Commitment t ∈ R_q^κ
    ///
    /// # Panics
    ///
    /// Panics if `s.len() != self.cols`.
    pub fn commit(&self, s: &[R]) -> AjtaiCommitment<R> {
        assert_eq!(
            s.len(),
            self.cols,
            "expected {} columns, got {}",
            self.cols,
            s.len()
        );

        // Compute t = A · s (matrix-vector product)
        // t[i] = Σ_j A[i][j] * s[j]
        let t: Vec<R> = (0..self.rows)
            .map(|i| {
                self.matrix[i]
                    .iter()
                    .zip(s)
                    .map(|(a_ij, s_j)| Ring::mul(a_ij, s_j))
                    .reduce(|acc, x| Ring::add(&acc, &x))
                    .unwrap_or_else(|| R::zero(self.d, self.q))
            })
            .collect();

        AjtaiCommitment {
            t,
            kappa: self.rows,
            d: self.d,
            q: self.q,
        }
    }
}

/// Convenience: convert coefficient-domain key to NTT domain
impl AjtaiKey<RingElement> {
    /// Convert to NTT domain for O(d) multiplications
    pub fn to_ntt(&self, ctx: &NttContext) -> AjtaiKey<NttRingElement> {
        AjtaiKey {
            rows: self.rows,
            cols: self.cols,
            d: self.d,
            q: self.q,
            matrix: ctx.forward_matrix(&self.matrix),
        }
    }
}

// ============================================================================
// Ajtai Commitment
// ============================================================================

/// Commitment output: t ∈ R_q^κ with metadata
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AjtaiCommitment<R: Ring> {
    /// Commitment vector t ∈ R_q^κ
    pub t: Vec<R>,
    /// Module rank κ
    pub kappa: usize,
    /// Ring dimension d
    pub d: usize,
    /// Modulus q
    pub q: u64,
}

// ============================================================================
// Decomposition-Based Interface (Hachi Workflow)
// ============================================================================

/// Commit to witnesses using base-b decomposition.
///
/// Computes t = A · G⁻¹(f₁, ..., fₗ) where G⁻¹ is base-b decomposition.
/// Coefficient-domain only (type-enforced).
///
/// # Arguments
///
/// * `key` - Ajtai commitment key with `cols = delta * witnesses.len()`
/// * `witnesses` - Vector of ring elements to commit
/// * `b` - Decomposition base
/// * `delta` - Number of decomposition digits (⌈log_b(q)⌉)
///
/// # Panics
///
/// Panics if `witnesses.len() * delta != key.cols`.
pub fn commit_decomposed(
    key: &AjtaiKey<RingElement>,
    witnesses: &[RingElement],
    b: u64,
    delta: usize,
) -> AjtaiCommitment<RingElement> {
    assert_eq!(
        witnesses.len() * delta,
        key.cols,
        "expected {} witnesses for {} columns with delta={}",
        key.cols / delta,
        key.cols,
        delta
    );

    let s: Vec<_> = witnesses
        .iter()
        .flat_map(|w| decompose_poly(w, b, delta))
        .collect();

    key.commit(&s)
}

/// Verify commitment against witnesses using decomposition.
///
/// Returns true if `commitment.t == A · G⁻¹(witnesses)`.
/// Coefficient-domain only (type-enforced).
///
/// # Arguments
///
/// * `key` - Ajtai commitment key
/// * `commitment` - Commitment to verify
/// * `witnesses` - Claimed witnesses
/// * `b` - Decomposition base
/// * `delta` - Number of decomposition digits
pub fn verify_decomposed(
    key: &AjtaiKey<RingElement>,
    commitment: &AjtaiCommitment<RingElement>,
    witnesses: &[RingElement],
    b: u64,
    delta: usize,
) -> bool {
    let recomputed = commit_decomposed(key, witnesses, b, delta);
    commitment.t == recomputed.t
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::decompose::recompose_poly;
    use crate::lattice::params::{COMPRESSED_K16, DILITHIUM_2, HACHI};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn commit_verify() {
        let mut rng = test_rng();

        let (d, q, b, delta) = (64, 65537, 16, 4);
        let num_witnesses = 2;

        let key: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 1, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        let commitment = commit_decomposed(&key, &witnesses, b, delta);
        assert!(verify_decomposed(&key, &commitment, &witnesses, b, delta));

        // Different witnesses should not verify
        let different: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();
        assert!(!verify_decomposed(&key, &commitment, &different, b, delta));
    }

    #[test]
    fn commit_raw_manual() {
        let mut rng = test_rng();

        // κ=2 rows, 4 columns (direct matrix-vector product, no decomposition)
        let (d, q) = (32, 65537);
        let key: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 2, 4, d, q);

        // Manually construct short vector s with small coefficients
        let s: Vec<_> = (0..key.cols)
            .map(|_| RingElement::random_bounded(&mut rng, d, q, 16))
            .collect();

        let commitment = key.commit(&s);

        // Verify dimensions
        assert_eq!(commitment.t.len(), 2);
        assert_eq!(commitment.kappa, 2);

        // Verify by recomputing manually: t[i] = Σ_j A[i][j] * s[j]
        for (i, t_i) in commitment.t.iter().enumerate() {
            let expected = key.matrix[i]
                .iter()
                .zip(&s)
                .map(|(a_ij, s_j)| a_ij.mul_schoolbook(s_j))
                .reduce(|acc, x| acc.add(&x))
                .unwrap_or_else(|| RingElement::zero(d, q));
            assert_eq!(*t_i, expected);
        }
    }

    #[test]
    fn commit_vs_decomposed() {
        // Verify that commit_decomposed() is equivalent to decompose + commit()
        let mut rng = test_rng();

        let (d, q, b, delta) = (64, 65537, 16, 4);
        let num_witnesses = 2;

        let key: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 1, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        // Via commit_decomposed()
        let c1 = commit_decomposed(&key, &witnesses, b, delta);

        // Via manual decompose + commit()
        let s: Vec<_> = witnesses
            .iter()
            .flat_map(|w| decompose_poly(w, b, delta))
            .collect();
        let c2 = key.commit(&s);

        assert_eq!(c1, c2);
    }

    #[test]
    fn commit_deterministic() {
        let mut rng = test_rng();

        let (d, q, b, delta) = (64, 65537, 16, 4);
        let num_witnesses = 2;

        let key: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 1, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        let c1 = commit_decomposed(&key, &witnesses, b, delta);
        let c2 = commit_decomposed(&key, &witnesses, b, delta);

        assert_eq!(c1, c2);
    }

    #[test]
    fn decompose_roundtrip() {
        let mut rng = test_rng();
        let (d, q, b, delta) = (HACHI.d, HACHI.q, HACHI.b, HACHI.delta);

        let f = RingElement::random(&mut rng, d, q);
        let decomposed = decompose_poly(&f, b, delta);
        let recomposed = recompose_poly(&decomposed, b);

        assert_eq!(f.coeffs, recomposed.coeffs);
    }

    #[test]
    fn small_coefficients() {
        let mut rng = test_rng();
        let (d, q, b, delta) = (HACHI.d, HACHI.q, HACHI.b, HACHI.delta);

        let f = RingElement::random(&mut rng, d, q);
        let decomposed = decompose_poly(&f, b, delta);

        for component in &decomposed {
            assert!(
                component.coeffs.iter().all(|&c| c < b),
                "coefficient exceeds base"
            );
        }
    }

    #[test]
    fn linearity_note() {
        // Note: commit_decomposed(f + g) ≠ commit_decomposed(f) + commit_decomposed(g)
        // in general because G⁻¹ is NOT linear (decomposition has carries).
        //
        // However, when coefficients of f + g stay below b = 16, no carry
        // occurs in base-16 decomposition, so: G⁻¹(f + g) = G⁻¹(f) + G⁻¹(g)

        let mut rng = test_rng();

        let (d, q, b, delta) = (64, 65537, 16, 4);
        let key: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 1, delta, d, q);

        // Coefficients in [0, 8), so f + g has coeffs in [0, 16) — no carry
        let f = RingElement::random_bounded(&mut rng, d, q, 8);
        let g = RingElement::random_bounded(&mut rng, d, q, 8);
        let f_plus_g = f.add(&g);

        let c_f = commit_decomposed(&key, std::slice::from_ref(&f), b, delta);
        let c_g = commit_decomposed(&key, std::slice::from_ref(&g), b, delta);
        let c_sum = commit_decomposed(&key, &[f_plus_g], b, delta);

        // c_f.t[0] + c_g.t[0] should equal c_sum.t[0] when no carry
        let manual_sum = c_f.t[0].add(&c_g.t[0]);

        assert_eq!(
            manual_sum.coeffs, c_sum.t[0].coeffs,
            "linearity holds when coefficients of f+g stay below b=16"
        );
    }

    #[test]
    fn hachi_params() {
        let mut rng = test_rng();
        let (d, q, b, delta) = (HACHI.d, HACHI.q, HACHI.b, HACHI.delta);
        let num_witnesses = 2;

        let key: AjtaiKey<RingElement> =
            AjtaiKey::random(&mut rng, HACHI.kappa, delta * num_witnesses, d, q);

        assert_eq!(key.rows, 1);
        assert_eq!(key.cols, 16); // 2 witnesses × 8 delta
        assert_eq!(key.d, 1024);

        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        let commitment = commit_decomposed(&key, &witnesses, b, delta);
        assert!(verify_decomposed(&key, &commitment, &witnesses, b, delta));
    }

    #[test]
    fn compressed_params() {
        let mut rng = test_rng();
        let (d, q, b, delta) = (
            COMPRESSED_K16.d,
            COMPRESSED_K16.q,
            COMPRESSED_K16.b,
            COMPRESSED_K16.delta,
        );
        let num_witnesses = 2;

        let key: AjtaiKey<RingElement> =
            AjtaiKey::random(&mut rng, COMPRESSED_K16.kappa, delta * num_witnesses, d, q);

        assert_eq!(key.d, 64);

        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        let commitment = commit_decomposed(&key, &witnesses, b, delta);
        assert!(verify_decomposed(&key, &commitment, &witnesses, b, delta));
    }

    // ========================================================================
    // NTT vs Schoolbook Tests
    // ========================================================================

    #[test]
    fn ntt_vs_schoolbook_commit() {
        // Use Dilithium params (NTT-friendly: q = 8380417 ≡ 1 mod 512)
        let mut rng = test_rng();
        let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);
        let ctx = NttContext::new(d, q, psi);

        // Coefficient domain key and short vector
        let key_coeff: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 2, 4, d, q);
        let s_coeff: Vec<_> = (0..4)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        // Commit in coefficient domain (schoolbook)
        let c_coeff = key_coeff.commit(&s_coeff);

        // Convert to NTT domain
        let key_ntt = key_coeff.to_ntt(&ctx);
        let s_ntt = ctx.forward_vec(&s_coeff);

        // Commit in NTT domain (pointwise)
        let c_ntt = key_ntt.commit(&s_ntt);

        // Compare by converting NTT result back to coefficient domain
        let c_ntt_back: Vec<RingElement> = c_ntt.t.iter().map(|e| ctx.inverse(e)).collect();

        assert_eq!(c_coeff.t, c_ntt_back);
    }

    #[test]
    fn ntt_commit_schoolbook_verify() {
        // Commit with NTT, verify with schoolbook — should agree
        let mut rng = test_rng();
        let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);
        let ctx = NttContext::new(d, q, psi);

        // Generate key and short vector in coefficient domain
        let key_coeff: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 2, 8, d, q);
        let s_coeff: Vec<_> = (0..8)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        // Commit in coefficient domain (schoolbook)
        let c_schoolbook = key_coeff.commit(&s_coeff);

        // Convert to NTT and commit
        let key_ntt = key_coeff.to_ntt(&ctx);
        let s_ntt = ctx.forward_vec(&s_coeff);
        let c_ntt = key_ntt.commit(&s_ntt);

        // Convert NTT commitment back
        let c_ntt_back: Vec<RingElement> = c_ntt.t.iter().map(|e| ctx.inverse(e)).collect();

        // Should be identical
        assert_eq!(c_schoolbook.t, c_ntt_back);
    }

    #[test]
    fn ntt_vs_schoolbook_decomposed() {
        // Use Dilithium params (NTT-friendly)
        let mut rng = test_rng();
        let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);
        let (b, delta) = (DILITHIUM_2.b, DILITHIUM_2.delta);
        let ctx = NttContext::new(d, q, psi);

        let num_witnesses = 2;
        let key_coeff: AjtaiKey<RingElement> =
            AjtaiKey::random(&mut rng, 1, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        // Commit using decomposition (coefficient domain)
        let c_schoolbook = commit_decomposed(&key_coeff, &witnesses, b, delta);

        // Decompose manually, convert to NTT, commit
        let s_coeff: Vec<_> = witnesses
            .iter()
            .flat_map(|w| decompose_poly(w, b, delta))
            .collect();
        let key_ntt = key_coeff.to_ntt(&ctx);
        let s_ntt = ctx.forward_vec(&s_coeff);
        let c_ntt = key_ntt.commit(&s_ntt);

        // Convert NTT result back
        let c_ntt_back: Vec<RingElement> = c_ntt.t.iter().map(|e| ctx.inverse(e)).collect();

        assert_eq!(c_schoolbook.t, c_ntt_back);
    }
}
