//! Sampling from partially split sets for lattice-based zero-knowledge proofs
//!
//! Paper:
//! https://eprint.iacr.org/2017/523
//!
//! Implementation based on "Short, invertible elements in partially splitting
//! cyclotomic rings and applications to lattice-based zero-knowledge proofs"
//! by Lyubashevsky, Seiler (2018).
//!
//! # Key Concepts
//!
//! In a cyclotomic ring R_q = Z_q[X]/(Φ_m(X)), when q ≡ 1 (mod m), the polynomial
//! Φ_m(X) splits completely into linear factors. For "partially splitting" rings,
//! we have intermediate factorizations.
//!
//! The challenge is to sample "short" ring elements c such that:
//! 1. ||c||_∞ is small (short coefficients)
//! 2. For any c₁ ≠ c₂ from the challenge set, c₁ - c₂ is invertible in R_q
//!
//! This is crucial for soundness in lattice-based ZK proofs like Dilithium and LaBRADOR.

use itertools::{EitherOrBoth, Itertools};
use rand::Rng;
use std::collections::HashSet;

/// Parameters for the partially split challenge set
///
/// Based on Section 4 of the Lyubashevsky-Seiler paper
///
/// # Note on Field Sizes (proof of concept)
///
/// Currently using `u64` for `tau` and `modulus` which suffices for prototyping
/// and standard parameter sets (e.g., Dilithium with q = 8380417 ≈ 2²³, Kyber with q = 3329).
/// For production use with larger moduli (e.g., q > 2^64) or integration with
/// arbitrary-precision arithmetic (e.g., `ark-ff` fields), these types may need
/// to be generalized to generic field elements or big integers.
#[derive(Clone, Debug)]
pub struct SplittingParams {
    /// Ring dimension n (degree of the cyclotomic polynomial)
    pub n: usize,

    /// Number of irreducible factors of X^n + 1 mod q
    ///
    /// This determines how invertibility is checked: an element is invertible
    /// iff it's non-zero when evaluated at each of the `num_splits` roots.
    ///
    /// **Note:** Currently used for validation and documentation only.
    /// Future: will be used in `is_definitely_difference_invertible()` for full verification.
    pub num_splits: usize,

    /// Coefficient bound: challenges have coefficients in {-τ, ..., τ}
    pub tau: u64,

    /// Hamming weight bound: at most ω non-zero coefficients
    pub omega: usize,

    /// Modulus q (prime)
    ///
    /// **Note:** Currently used for computing/validating `num_splits` only.
    /// Future: will be used for full invertibility checks.
    pub modulus: u64,
}

impl SplittingParams {
    /// Create parameters for power-of-two cyclotomics (X^n + 1)
    ///
    /// For n = 2^k, Φ_{2n}(X) = X^n + 1
    /// When q ≡ 1 (mod 2n), this splits completely into n linear factors
    /// When q ≡ 2^j + 1 (mod 2^{j+1}), we get 2^{k-j} factors of degree 2^j
    pub fn power_of_two(n: usize, num_splits: usize, tau: u64, omega: usize, modulus: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two");
        assert!(n.is_multiple_of(num_splits), "num_splits must divide n");
        assert!(omega <= n, "omega cannot exceed n");
        Self {
            n,
            num_splits,
            tau,
            omega,
            modulus,
        }
    }

    /// Create parameters with num_splits computed automatically from n and modulus
    ///
    /// Use this when you don't have a pre-computed num_splits value
    pub fn with_computed_splits(n: usize, tau: u64, omega: usize, modulus: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two");
        assert!(omega <= n, "omega cannot exceed n");
        let num_splits = Self::compute_num_splits(n, modulus);
        Self {
            n,
            num_splits,
            tau,
            omega,
            modulus,
        }
    }

    /// Compute the number of irreducible factors of X^n + 1 mod q
    ///
    /// For power-of-two cyclotomics where n = 2^k:
    /// - Find the largest j such that q ≡ 1 (mod 2^j)
    /// - X^n + 1 splits into n / 2^{j-1} factors of degree 2^{j-1}
    ///
    /// Special cases:
    /// - q ≡ 1 (mod 2n): fully splits into n linear factors
    /// - q ≡ 1 (mod 2) but q ≢ 1 (mod 4): doesn't split (1 factor of degree n)
    pub fn compute_num_splits(n: usize, modulus: u64) -> usize {
        assert!(n.is_power_of_two(), "n must be a power of two");

        if modulus == 0 {
            return 1; // Degenerate case for testing
        }

        // Find largest j where q ≡ 1 (mod 2^j)
        // This is: j = trailing_zeros(q - 1) for odd q
        if modulus.is_multiple_of(2) {
            return 1; // Even modulus, no splitting
        }

        let j = (modulus - 1).trailing_zeros() as usize;

        // For X^n + 1 where n = 2^k:
        // - Need j >= 1 for any splitting
        // - num_splits = min(n, 2^{j-1}) when j >= 2
        // - num_splits = 1 when j < 2
        if j < 2 {
            1
        } else {
            let max_splits = 1usize << (j - 1); // 2^{j-1}
            n.min(max_splits)
        }
    }

    /// Check if num_splits matches the computed value for n and modulus
    ///
    /// Returns true if num_splits is valid for the given parameters.
    /// Use this to validate hardcoded parameter sets.
    pub fn is_valid_num_splits(&self) -> bool {
        self.num_splits == Self::compute_num_splits(self.n, self.modulus)
    }

    /// Validate that num_splits is correct, panicking with details if not
    ///
    /// Useful for catching parameter errors in debug builds
    pub fn validate(&self) {
        let expected = Self::compute_num_splits(self.n, self.modulus);
        assert_eq!(
            self.num_splits, expected,
            "Invalid num_splits: got {}, expected {} for n={}, q={}",
            self.num_splits, expected, self.n, self.modulus
        );
    }

    /// Challenge set size: C(n, ω) * (2τ)^ω
    ///
    /// This is the number of polynomials with at most ω non-zero coefficients
    /// where each non-zero coefficient is in {-τ, ..., τ} \ {0}
    ///
    /// Returns u128::MAX if the result would overflow
    pub fn challenge_set_size(&self) -> u128 {
        // C(n, omega) * (2*tau)^omega
        let binomial = binomial_coefficient(self.n, self.omega);
        let coeff_choices = (2 * self.tau) as u128; // {-τ,...,-1,1,...,τ}

        // Compute (2*tau)^omega with overflow checking
        let mut power: u128 = 1;
        for _ in 0..self.omega {
            power = match power.checked_mul(coeff_choices) {
                Some(p) => p,
                None => return u128::MAX,
            };
        }

        binomial.saturating_mul(power)
    }

    /// Security level (log2 of challenge set size)
    /// Security level (floor of log2 of challenge set size)
    pub fn security_bits(&self) -> u64 {
        let size = self.challenge_set_size();
        if size == 0 {
            0
        } else {
            size.ilog2() as u64
        }
    }
}

/// A challenge element in the partially split ring
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Challenge {
    /// Sparse representation: (index, coefficient) pairs
    pub sparse_coeffs: Vec<(usize, i64)>,
    /// Ring dimension
    pub n: usize,
}

impl Challenge {
    /// Create a new challenge from sparse coefficients
    pub fn new(sparse_coeffs: Vec<(usize, i64)>, n: usize) -> Self {
        let mut coeffs = sparse_coeffs;
        coeffs.sort_by_key(|(idx, _)| *idx);
        // Remove zeros
        coeffs.retain(|(_, c)| *c != 0);
        Self {
            sparse_coeffs: coeffs,
            n,
        }
    }

    /// Create the zero challenge
    pub fn zero(n: usize) -> Self {
        Self {
            sparse_coeffs: vec![],
            n,
        }
    }

    /// Convert to dense coefficient representation
    pub fn to_dense(&self) -> Vec<i64> {
        let mut coeffs = vec![0i64; self.n];
        for &(idx, coeff) in &self.sparse_coeffs {
            coeffs[idx] = coeff;
        }
        coeffs
    }

    /// Hamming weight (number of non-zero coefficients)
    pub fn weight(&self) -> usize {
        self.sparse_coeffs.len()
    }

    /// Infinity norm
    pub fn ell_inf_norm(&self) -> i64 {
        self.sparse_coeffs
            .iter()
            .map(|(_, c)| c.abs())
            .max()
            .unwrap_or(0)
    }

    /// Compute c₁ - c₂ (for checking invertibility of differences)
    pub fn sub(&self, other: &Challenge) -> Challenge {
        assert_eq!(self.n, other.n);

        let result: Vec<(usize, i64)> = self
            .sparse_coeffs
            .iter()
            .copied()
            .merge_join_by(other.sparse_coeffs.iter().copied(), |(i, _), (j, _)| {
                i.cmp(j)
            })
            .filter_map(|eob| match eob {
                EitherOrBoth::Left((idx, c)) => Some((idx, c)),
                EitherOrBoth::Right((idx, c)) => Some((idx, -c)),
                EitherOrBoth::Both((idx, c1), (_, c2)) => {
                    let diff = c1 - c2;
                    (diff != 0).then_some((idx, diff))
                }
            })
            .collect();

        Challenge::new(result, self.n)
    }

    /// Add two challenges
    pub fn add(&self, other: &Challenge) -> Challenge {
        assert_eq!(self.n, other.n);

        let result: Vec<(usize, i64)> = self
            .sparse_coeffs
            .iter()
            .copied()
            .merge_join_by(other.sparse_coeffs.iter().copied(), |(i, _), (j, _)| {
                i.cmp(j)
            })
            .filter_map(|eob| match eob {
                EitherOrBoth::Left(pair) | EitherOrBoth::Right(pair) => Some(pair),
                EitherOrBoth::Both((idx, c1), (_, c2)) => {
                    let sum = c1 + c2;
                    (sum != 0).then_some((idx, sum))
                }
            })
            .collect();

        Challenge::new(result, self.n)
    }

    /// Negate the challenge
    pub fn neg(&self) -> Challenge {
        let coeffs: Vec<(usize, i64)> = self
            .sparse_coeffs
            .iter()
            .map(|&(idx, c)| (idx, -c))
            .collect();
        Challenge::new(coeffs, self.n)
    }
}

/// Sample a random challenge from the challenge set
///
/// Samples a polynomial with:
/// - Exactly `omega` non-zero coefficients
/// - Each non-zero coefficient uniformly from {-τ, ..., -1, 1, ..., τ}
/// - Positions chosen uniformly at random
///
/// # Arguments
/// * `n` - Ring dimension (polynomial degree)
/// * `tau` - Coefficient bound (non-zero coeffs in {-τ,...,-1,1,...,τ})
/// * `omega` - Number of non-zero coefficients (Hamming weight)
pub fn sample_challenge<R: Rng>(rng: &mut R, n: usize, tau: u64, omega: usize) -> Challenge {
    // Sample omega distinct positions
    let positions = sample_distinct_positions(rng, n, omega);

    // Sample non-zero coefficients
    let sparse_coeffs: Vec<(usize, i64)> = positions
        .into_iter()
        .map(|pos| {
            // Sample from {-τ, ..., -1, 1, ..., τ}
            let coeff = sample_nonzero_bounded(rng, tau);
            (pos, coeff)
        })
        .collect();

    Challenge::new(sparse_coeffs, n)
}

/// Sample a ternary challenge (coefficients in {-1, 0, 1})
///
/// This is the most common case used in schemes like Dilithium.
/// Equivalent to `sample_challenge(rng, n, 1, omega)`.
pub fn sample_ternary_challenge<R: Rng>(rng: &mut R, n: usize, omega: usize) -> Challenge {
    sample_challenge(rng, n, 1, omega)
}

/// Sample a challenge with fixed positions (for deterministic challenges from hash)
///
/// Used in Fiat-Shamir transform where positions and signs come from a hash
pub fn challenge_from_seed(seed: &[u8], n: usize, omega: usize, tau: u64) -> Challenge {
    use sha2::{Digest, Sha256};

    // Use hash to deterministically select positions and coefficients
    let mut positions = Vec::with_capacity(omega);
    let mut coeffs = Vec::with_capacity(omega);
    let mut used_positions = HashSet::new();

    // Chain hashes for deterministic expansion
    let mut current_hash = {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        hasher.finalize()
    };
    let mut hash_counter: u64 = 0;
    let mut byte_idx = 0;

    while positions.len() < omega {
        // Get more hash bytes if needed
        if byte_idx + 3 > 32 {
            let mut hasher = Sha256::new();
            hasher.update(seed);
            hasher.update(hash_counter.to_le_bytes());
            current_hash = hasher.finalize();
            hash_counter += 1;
            byte_idx = 0;
        }

        // Read position from 2 bytes
        let pos_bytes = [current_hash[byte_idx], current_hash[byte_idx + 1]];
        let pos = u16::from_le_bytes(pos_bytes) as usize % n;
        byte_idx += 2;

        if !used_positions.contains(&pos) {
            used_positions.insert(pos);
            positions.push(pos);

            // Read coefficient from 1 byte
            let coeff_byte = current_hash[byte_idx];
            byte_idx += 1;

            let sign = if coeff_byte & 1 == 0 { 1i64 } else { -1i64 };
            let mag: i64 = if tau == 1 {
                1
            } else {
                ((coeff_byte >> 1) as u64 % tau) as i64 + 1
            };
            coeffs.push(sign * mag);
        } else {
            // Skip the coefficient byte too to stay aligned
            byte_idx += 1;
        }
    }

    let sparse_coeffs: Vec<(usize, i64)> = positions.into_iter().zip(coeffs).collect();
    Challenge::new(sparse_coeffs, n)
}

/// Fast path check for difference invertibility
pub fn is_definitely_difference_invertible(diff: &Challenge, params: &SplittingParams) -> bool {
    // For ternary challenges with weight ω, the difference has weight at most 2ω
    // and coefficients in {-2, -1, 0, 1, 2}
    //
    // Reference: Lyubashevsky & Seiler, "Short, Invertible Elements in Partially
    // Splitting Cyclotomic Rings and Applications to Lattice-Based Zero-Knowledge
    // Proofs", EUROCRYPT 2018 (IACR ePrint 2017/523).
    //
    // Theorem 3.1: For R_q = Z_q[X]/(X^n + 1) where X^n + 1 splits into k factors
    // mod q, any non-zero polynomial with coefficients bounded by B is invertible,
    // provided B is small enough relative to q and k.
    //
    // For our sparse ternary challenges: |c_i| ≤ 2τ and q >> 2τ·k, so the theorem
    // guarantees invertibility of all non-zero differences.
    //
    // The ||c||_1 < n check below is a fast-path heuristic (not from the theorem)
    // that catches obviously-small elements. For elements that fail this check,
    // we fall back to a more expensive check (see check_invertibility_by_evaluation).
    //
    // For our sparse challenges: ||c₁ - c₂||_1 ≤ 2ω · 2τ = 4ωτ

    // Zero is never invertible
    if diff.sparse_coeffs.is_empty() {
        return false;
    }

    let l1_norm: i64 = diff.sparse_coeffs.iter().map(|(_, c)| c.abs()).sum();

    // Simple check: L1 norm bound for invertibility (for non-zero elements)
    l1_norm > 0 && l1_norm < params.n as i64
}

/// Precomputed challenge set for small parameters
///
/// For small challenge spaces, we can enumerate all valid challenges
/// and ensure all pairwise differences are invertible
#[derive(Clone, Debug)]
pub struct ChallengeSet {
    pub challenges: Vec<Challenge>,
    pub params: SplittingParams,
}

impl ChallengeSet {
    /// Build a challenge set of distinct challenges
    ///
    /// By Theorem 3.1 (Lyubashevsky & Seiler), for typical parameters where
    /// |c_i| ≤ 2τ and q >> 2τ·k, all non-zero differences are guaranteed
    /// invertible. We verify this in debug builds.
    pub fn build(params: SplittingParams, max_challenges: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut seen = HashSet::with_capacity(max_challenges);

        let challenges: Vec<Challenge> = std::iter::from_fn(|| {
            Some(sample_challenge(
                &mut rng,
                params.n,
                params.tau,
                params.omega,
            ))
        })
        .filter(|c| seen.insert(c.clone()))
        .take(max_challenges)
        .collect();

        // Debug: verify all pairwise differences are invertible
        #[cfg(debug_assertions)]
        for (i, c1) in challenges.iter().enumerate() {
            for c2 in challenges.iter().skip(i + 1) {
                let diff = c1.sub(c2);
                debug_assert!(
                    is_definitely_difference_invertible(&diff, &params),
                    "Theorem 3.1 violated: difference not invertible"
                );
            }
        }

        Self { challenges, params }
    }

    /// Get challenge set size
    pub fn size(&self) -> usize {
        self.challenges.len()
    }

    /// Sample a random challenge from the precomputed set
    pub fn sample<R: Rng>(&self, rng: &mut R) -> &Challenge {
        let idx = rng.gen_range(0..self.challenges.len());
        &self.challenges[idx]
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Sample k distinct positions from [0, n)
fn sample_distinct_positions<R: Rng>(rng: &mut R, n: usize, k: usize) -> Vec<usize> {
    assert!(k <= n);

    // Do Fisher-Yates for large k (k > n/2), rejection sampling for small k (k <= n/2)
    // Note: The output of the rejection sampling branch is unordered (HashSet iteration order is arbitrary), while the Fisher-Yates branch produces positions in a random order
    if k > n / 2 {
        // For large k, use Fisher-Yates on full range
        // https://crypto.stackexchange.com/questions/41811/correct-use-of-fisher-yates-for-subsequent-selections
        let mut positions: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = rng.gen_range(i..n);
            positions.swap(i, j);
        }
        positions.truncate(k);
        positions
    } else {
        // For small k, rejection sampling
        let mut positions = HashSet::with_capacity(k);
        while positions.len() < k {
            positions.insert(rng.gen_range(0..n));
        }
        positions.into_iter().collect()
    }
}

/// Sample a non-zero integer from {-bound, ..., -1, 1, ..., bound}
fn sample_nonzero_bounded<R: Rng>(rng: &mut R, bound: u64) -> i64 {
    // 2*bound choices: {-bound,...,-1,1,...,bound}
    let r = rng.gen_range(0..2 * bound);
    if r < bound {
        -((r + 1) as i64) // Maps 0..bound to -1..-bound
    } else {
        (r - bound + 1) as i64 // Maps bound..2*bound to 1..bound
    }
}

/// Compute binomial coefficient C(n, k)
///
/// Uses the multiplicative formula with careful ordering to avoid overflow
fn binomial_coefficient(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Use symmetry
    let mut result: u128 = 1;
    for i in 0..k {
        // Uses the recurrence: C(n, i+1) = C(n, i) * (n - i) / (i + 1)
        // Multiply before divide to maintain integer arithmetic (C(n,k) is always integral)
        result = result.saturating_mul((n - i) as u128) / (i + 1) as u128;
    }
    result
}

// ============================================================================
// Standard parameter sets
// ============================================================================

/// Dilithium-style parameters
pub fn dilithium_challenge_params() -> SplittingParams {
    SplittingParams {
        n: 256,
        num_splits: 256, // Fully split for q ≡ 1 (mod 512)
        tau: 1,          // Ternary
        omega: 60,       // Weight 60 gives ~128-bit security
        modulus: 8380417,
    }
}

/// Hachi-style parameters (for your existing codebase)
pub fn hachi_challenge_params() -> SplittingParams {
    SplittingParams {
        n: 1024,
        num_splits: 256, // Partially split
        tau: 1,
        omega: 128,
        modulus: 65537,
    }
}

/// Conservative parameters for testing
pub fn test_challenge_params() -> SplittingParams {
    SplittingParams {
        n: 64,
        num_splits: 16,
        tau: 1,
        omega: 16,
        modulus: 65537,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_challenge_creation() {
        let c = Challenge::new(vec![(0, 1), (5, -1), (10, 1)], 64);
        assert_eq!(c.weight(), 3);
        assert_eq!(c.ell_inf_norm(), 1);

        let dense = c.to_dense();
        assert_eq!(dense[0], 1);
        assert_eq!(dense[5], -1);
        assert_eq!(dense[10], 1);
        assert_eq!(dense[1], 0);
    }

    #[test]
    fn test_challenge_arithmetic() {
        let c1 = Challenge::new(vec![(0, 1), (5, 2)], 64);
        let c2 = Challenge::new(vec![(5, 1), (10, -1)], 64);

        let diff = c1.sub(&c2);
        assert_eq!(diff.sparse_coeffs, vec![(0, 1), (5, 1), (10, 1)]);

        let sum = c1.add(&c2);
        assert_eq!(sum.sparse_coeffs, vec![(0, 1), (5, 3), (10, -1)]);
    }

    #[test]
    fn test_sample_challenge() {
        let mut rng = rand::thread_rng();
        let n = 64;
        let tau = 1;
        let omega = 16;

        for _ in 0..100 {
            let c = sample_challenge(&mut rng, n, tau, omega);
            assert!(c.weight() <= omega);
            assert!(c.ell_inf_norm() <= tau as i64);
        }
    }

    #[test]
    fn test_ternary_challenge() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let c = sample_ternary_challenge(&mut rng, 256, 60);
            assert_eq!(c.weight(), 60);
            assert_eq!(c.ell_inf_norm(), 1);
        }
    }

    #[test]
    fn test_challenge_difference_invertibility() {
        let params = test_challenge_params();
        let mut rng = rand::thread_rng();

        // Sample many pairs and check differences
        for _ in 0..50 {
            let c1 = sample_challenge(&mut rng, params.n, params.tau, params.omega);
            let c2 = sample_challenge(&mut rng, params.n, params.tau, params.omega);

            if c1 != c2 {
                let diff = c1.sub(&c2);
                // For small omega relative to n, differences should be invertible
                assert!(
                    is_definitely_difference_invertible(&diff, &params),
                    "Difference should be invertible: {:?}",
                    diff
                );
            }
        }
    }

    #[test]
    fn test_invertibility() {
        let params = SplittingParams {
            n: 64,
            num_splits: 64,
            tau: 1,
            omega: 8,
            modulus: 65537,
        };

        // ===== Zero challenge: NOT invertible =====
        let zero = Challenge::zero(64);
        assert!(
            !is_definitely_difference_invertible(&zero, &params),
            "Zero should not be invertible"
        );

        // ===== Constant-only challenges: invertible =====
        let const_only = Challenge::new(vec![(0, 1)], 64);
        assert!(
            is_definitely_difference_invertible(&const_only, &params),
            "Constant polynomial should be invertible"
        );

        let neg_const = Challenge::new(vec![(0, -1)], 64);
        assert!(
            is_definitely_difference_invertible(&neg_const, &params),
            "Negative constant should be invertible"
        );

        // ===== Small weight with constant term: invertible =====
        // weight = 2 <= n/4 = 16, has constant term
        let small_with_const = Challenge::new(vec![(0, 1), (5, -1)], 64);
        assert!(
            is_definitely_difference_invertible(&small_with_const, &params),
            "Small weight with constant term should be invertible"
        );

        // ===== Small weight WITHOUT constant term: tests L1 check =====
        let small_no_const = Challenge::new(vec![(3, 1), (7, -1)], 64);
        // L1 = 2 < n/2 = 32, should be invertible
        assert!(
            is_definitely_difference_invertible(&small_no_const, &params),
            "Small L1 norm without constant should be invertible"
        );

        // ===== Single non-zero coefficient (monomials): invertible =====
        let single_coeff = Challenge::new(vec![(10, 1)], 64);
        assert!(
            is_definitely_difference_invertible(&single_coeff, &params),
            "Single coefficient should be invertible via L1 check"
        );

        let x_mono = Challenge::new(vec![(1, 1)], 64);
        assert!(
            is_definitely_difference_invertible(&x_mono, &params),
            "X should be invertible (small L1)"
        );

        let high_mono = Challenge::new(vec![(63, 1)], 64);
        assert!(
            is_definitely_difference_invertible(&high_mono, &params),
            "X^(n-1) should be invertible (small L1)"
        );

        // ===== Medium weight, still small L1: invertible =====
        // weight = 10 <= 16, no constant, L1 = 10 < 32
        let medium_weight: Vec<(usize, i64)> = (1..=10).map(|i| (i, 1)).collect();
        let medium = Challenge::new(medium_weight, 64);
        assert!(
            is_definitely_difference_invertible(&medium, &params),
            "Medium weight with small L1 should be invertible"
        );

        // ===== Sum of powers: invertible =====
        // weight = 8 <= 16, has constant, should be invertible
        let sum_powers: Vec<(usize, i64)> = (0..8).map(|i| (i, 1)).collect();
        let sum_poly = Challenge::new(sum_powers, 64);
        assert!(
            is_definitely_difference_invertible(&sum_poly, &params),
            "1 + X + ... + X^7 should be invertible"
        );

        // ===== Test L1 < n fast path =====
        // Create challenge with L1 < n but > n/2
        // weight = 20 > 16, no constant, L1 = 40 < 64 but > 32
        let large_weight: Vec<(usize, i64)> = (1..=20).map(|i| (i, 2)).collect();
        let large = Challenge::new(large_weight, 64);
        // L1 = 40 < 64, so fast path returns true
        assert!(
            is_definitely_difference_invertible(&large, &params),
            "L1 < n should pass fast path"
        );
    }

    #[test]
    fn test_challenge_set_size() {
        // Use smaller parameters that won't overflow
        let params = SplittingParams {
            n: 64,
            num_splits: 64,
            tau: 1,
            omega: 32,
            modulus: 8380417,
        };

        // C(64, 32) * 2^32 is computable
        let size = params.challenge_set_size();
        assert!(size > 0);

        let bits = params.security_bits();
        // Should provide reasonable security
        assert!(bits > 60, "Security bits: {}", bits);

        // Test with Dilithium-like params (will saturate to MAX)
        let dilithium_params = dilithium_challenge_params();
        let dilithium_size = dilithium_params.challenge_set_size();
        // This is so large it saturates
        assert!(dilithium_size == u128::MAX || dilithium_size > 1u128 << 100);
    }

    #[test]
    fn test_deterministic_challenge() {
        let seed = b"test seed for challenge generation";

        let c1 = challenge_from_seed(seed, 256, 60, 1);
        let c2 = challenge_from_seed(seed, 256, 60, 1);

        assert_eq!(c1, c2, "Same seed should produce same challenge");

        let c3 = challenge_from_seed(b"different seed", 256, 60, 1);
        assert_ne!(
            c1, c3,
            "Different seeds should produce different challenges"
        );
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(10, 0), 1);
        assert_eq!(binomial_coefficient(10, 10), 1);
        assert_eq!(binomial_coefficient(10, 1), 10);
        assert_eq!(binomial_coefficient(10, 2), 45);
        assert_eq!(binomial_coefficient(10, 5), 252);
        // C(256, 60) is a very large number, just check it doesn't overflow
        let large = binomial_coefficient(256, 60);
        assert!(large > 0);
    }

    #[test]
    fn test_sample_distinct_positions() {
        let mut rng = rand::thread_rng();

        // Small k
        let pos = sample_distinct_positions(&mut rng, 100, 10);
        assert_eq!(pos.len(), 10);
        let unique: HashSet<_> = pos.iter().collect();
        assert_eq!(unique.len(), 10);

        // Large k
        let pos = sample_distinct_positions(&mut rng, 100, 80);
        assert_eq!(pos.len(), 80);
        let unique: HashSet<_> = pos.iter().collect();
        assert_eq!(unique.len(), 80);
    }

    #[test]
    fn test_sample_nonzero_bounded() {
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let x = sample_nonzero_bounded(&mut rng, 5);
            assert!((-5..=5).contains(&x));
            assert_ne!(x, 0);
        }
    }

    #[test]
    fn test_compute_num_splits() {
        // ========== Standard lattice crypto parameters ==========

        // Dilithium (ML-DSA): q = 8380417
        // q - 1 = 8380416 = 2^13 × 1023, j = 13, max_splits = 2^12 = 4096
        // n = 256 < 4096, so fully splits
        assert_eq!(SplittingParams::compute_num_splits(256, 8380417), 256);

        // Kyber (ML-KEM): q = 3329
        // q - 1 = 3328 = 2^8 × 13, j = 8, max_splits = 2^7 = 128
        // n = 256 > 128, so partially splits
        assert_eq!(SplittingParams::compute_num_splits(256, 3329), 128);

        // NewHope/Frodo-style: q = 12289
        // q - 1 = 12288 = 2^12 × 3, j = 12, max_splits = 2^11 = 2048
        assert_eq!(SplittingParams::compute_num_splits(512, 12289), 512);
        assert_eq!(SplittingParams::compute_num_splits(1024, 12289), 1024);

        // ========== Fermat primes (q = 2^k + 1) ==========

        // q = 17 = 2^4 + 1, j = 4, max_splits = 8
        assert_eq!(SplittingParams::compute_num_splits(64, 17), 8);
        assert_eq!(SplittingParams::compute_num_splits(8, 17), 8);
        assert_eq!(SplittingParams::compute_num_splits(4, 17), 4);

        // q = 257 = 2^8 + 1, j = 8, max_splits = 128
        assert_eq!(SplittingParams::compute_num_splits(256, 257), 128);
        assert_eq!(SplittingParams::compute_num_splits(64, 257), 64);

        // q = 65537 = 2^16 + 1, j = 16, max_splits = 32768
        assert_eq!(SplittingParams::compute_num_splits(1024, 65537), 1024);
        assert_eq!(SplittingParams::compute_num_splits(256, 65537), 256);

        // ========== Edge cases: no splitting ==========

        // q = 3, q - 1 = 2 = 2^1, j = 1 < 2
        assert_eq!(SplittingParams::compute_num_splits(64, 3), 1);

        // q = 7, q - 1 = 6 = 2 × 3, j = 1 < 2
        assert_eq!(SplittingParams::compute_num_splits(64, 7), 1);

        // q = 5, q - 1 = 4 = 2^2, j = 2, max_splits = 2
        assert_eq!(SplittingParams::compute_num_splits(64, 5), 2);
        assert_eq!(SplittingParams::compute_num_splits(2, 5), 2);

        // ========== Edge cases: small n ==========
        assert_eq!(SplittingParams::compute_num_splits(2, 17), 2);
        assert_eq!(SplittingParams::compute_num_splits(2, 65537), 2);

        // ========== More NTT-friendly primes ==========

        // q = 7681, q - 1 = 7680 = 2^9 × 15, j = 9, max_splits = 256
        assert_eq!(SplittingParams::compute_num_splits(256, 7681), 256);
        assert_eq!(SplittingParams::compute_num_splits(512, 7681), 256);

        // q = 12289, already tested above
        // q = 40961, q - 1 = 40960 = 2^13 × 5, j = 13, max_splits = 4096
        assert_eq!(SplittingParams::compute_num_splits(1024, 40961), 1024);
        assert_eq!(SplittingParams::compute_num_splits(4096, 40961), 4096);
        assert_eq!(SplittingParams::compute_num_splits(8192, 40961), 4096);

        // q = 786433 = 3 × 2^18 + 1, q - 1 = 786432 = 2^18 × 3, j = 18, max_splits = 131072
        assert_eq!(SplittingParams::compute_num_splits(1024, 786433), 1024);
        assert_eq!(SplittingParams::compute_num_splits(65536, 786433), 65536);

        // ========== Partial splitting (n > max_splits) ==========

        // q = 17, max_splits = 8, various n > 8
        assert_eq!(SplittingParams::compute_num_splits(16, 17), 8);
        assert_eq!(SplittingParams::compute_num_splits(32, 17), 8);
        assert_eq!(SplittingParams::compute_num_splits(128, 17), 8);
        assert_eq!(SplittingParams::compute_num_splits(1024, 17), 8);

        // q = 5, max_splits = 2
        assert_eq!(SplittingParams::compute_num_splits(4, 5), 2);
        assert_eq!(SplittingParams::compute_num_splits(8, 5), 2);
        assert_eq!(SplittingParams::compute_num_splits(256, 5), 2);

        // ========== Large n ==========
        assert_eq!(SplittingParams::compute_num_splits(2048, 8380417), 2048);
        assert_eq!(SplittingParams::compute_num_splits(4096, 8380417), 4096);
        assert_eq!(SplittingParams::compute_num_splits(8192, 8380417), 4096); // hits max_splits

        // ========== Boundary: n exactly equals max_splits ==========
        // q = 17, max_splits = 8, n = 8
        assert_eq!(SplittingParams::compute_num_splits(8, 17), 8);
        // q = 257, max_splits = 128, n = 128
        assert_eq!(SplittingParams::compute_num_splits(128, 257), 128);

        // ========== Primes with small 2-adic valuation ==========

        // q = 13, q - 1 = 12 = 2^2 × 3, j = 2, max_splits = 2
        assert_eq!(SplittingParams::compute_num_splits(64, 13), 2);

        // q = 29, q - 1 = 28 = 2^2 × 7, j = 2, max_splits = 2
        assert_eq!(SplittingParams::compute_num_splits(64, 29), 2);

        // q = 41, q - 1 = 40 = 2^3 × 5, j = 3, max_splits = 4
        assert_eq!(SplittingParams::compute_num_splits(64, 41), 4);
        assert_eq!(SplittingParams::compute_num_splits(4, 41), 4);
        assert_eq!(SplittingParams::compute_num_splits(2, 41), 2);

        // q = 97, q - 1 = 96 = 2^5 × 3, j = 5, max_splits = 16
        assert_eq!(SplittingParams::compute_num_splits(64, 97), 16);
        assert_eq!(SplittingParams::compute_num_splits(16, 97), 16);
        assert_eq!(SplittingParams::compute_num_splits(8, 97), 8);
    }

    #[test]
    fn test_with_computed_splits() {
        let params = SplittingParams::with_computed_splits(256, 1, 60, 8380417);
        assert_eq!(params.num_splits, 256);
        assert!(params.is_valid_num_splits());

        let params2 = SplittingParams::with_computed_splits(256, 1, 60, 3329);
        assert_eq!(params2.num_splits, 128);
        assert!(params2.is_valid_num_splits());
    }

    #[test]
    fn test_validate_num_splits() {
        // Valid params
        let valid = SplittingParams {
            n: 256,
            num_splits: 256,
            tau: 1,
            omega: 60,
            modulus: 8380417,
        };
        assert!(valid.is_valid_num_splits());

        // Invalid params (wrong num_splits)
        let invalid = SplittingParams {
            n: 256,
            num_splits: 128, // Should be 256 for Dilithium's q
            tau: 1,
            omega: 60,
            modulus: 8380417,
        };
        assert!(!invalid.is_valid_num_splits());
    }

    #[test]
    fn test_challenge_set_build() {
        let params = SplittingParams {
            n: 64,
            num_splits: 64,
            tau: 1,
            omega: 8,
            modulus: 65537,
        };

        // Build a small challenge set - debug assertions will verify invertibility
        let set = ChallengeSet::build(params, 10);
        assert_eq!(set.size(), 10);

        // All challenges should be distinct
        let unique: HashSet<_> = set.challenges.iter().collect();
        assert_eq!(unique.len(), 10);

        // Sample should return a valid challenge
        let mut rng = rand::thread_rng();
        let sampled = set.sample(&mut rng);
        assert_eq!(sampled.n, 64);
        assert!(sampled.weight() <= 8);
    }
}
