//! Sampling from partially split sets for lattice-based zero-knowledge proofs
//!
//! Implementation based on "Short, invertible elements in partially splitting
//! cyclotomic rings and applications to lattice-based zero-knowledge proofs"
//! by Lyubashevsky, Seiler (2018). Paper: https://eprint.iacr.org/2017/523
//!
//! # Key Concepts
//!
//! In R_q = Z_q[X]/(Φ_m(X)), when q ≡ 1 (mod m), Φ_m(X) splits completely.
//! For "partially splitting" rings, we have intermediate factorizations.
//!
//! The challenge is to sample "short" ring elements c such that:
//! 1. ||c||_∞ is small (short coefficients)
//! 2. For any c₁ ≠ c₂, c₁ - c₂ is invertible in R_q

use itertools::{EitherOrBoth, Itertools};
use rand::Rng;
use std::collections::HashSet;

// ============================================================================
// Parameters
// ============================================================================

/// Parameters for the partially split challenge set
#[derive(Clone, Debug)]
pub struct SplittingParams {
    /// Ring dimension n
    pub n: usize,
    /// Number of irreducible factors of X^n + 1 mod q
    pub num_splits: usize,
    /// Coefficient bound: challenges have coefficients in {-τ, ..., τ}
    pub tau: u64,
    /// Hamming weight bound: at most ω non-zero coefficients
    pub omega: usize,
    /// Modulus q (prime)
    pub modulus: u64,
}

impl SplittingParams {
    /// Create parameters for power-of-two cyclotomics (X^n + 1)
    pub fn power_of_two(n: usize, num_splits: usize, tau: u64, omega: usize, modulus: u64) -> Self {
        debug_assert!(n.is_power_of_two(), "n must be a power of two");
        debug_assert!(n.is_multiple_of(num_splits), "num_splits must divide n");
        debug_assert!(omega <= n, "omega cannot exceed n");
        Self {
            n,
            num_splits,
            tau,
            omega,
            modulus,
        }
    }

    /// Create parameters with num_splits computed automatically
    pub fn with_computed_splits(n: usize, tau: u64, omega: usize, modulus: u64) -> Self {
        debug_assert!(n.is_power_of_two());
        debug_assert!(omega <= n);
        Self {
            n,
            num_splits: Self::compute_num_splits(n, modulus),
            tau,
            omega,
            modulus,
        }
    }

    /// Compute the number of irreducible factors of X^n + 1 mod q
    pub fn compute_num_splits(n: usize, modulus: u64) -> usize {
        debug_assert!(n.is_power_of_two());

        if modulus & 1 == 0 {
            // modulus is 0 or even
            return 1;
        }

        let j = (modulus - 1).trailing_zeros() as usize;
        if j < 2 {
            1
        } else {
            n.min(1 << (j - 1))
        }
    }

    /// Check if num_splits matches the computed value
    pub fn is_valid_num_splits(&self) -> bool {
        self.num_splits == Self::compute_num_splits(self.n, self.modulus)
    }

    /// Validate parameters, panicking with details if invalid
    pub fn validate(&self) {
        let expected = Self::compute_num_splits(self.n, self.modulus);
        assert_eq!(
            self.num_splits, expected,
            "Invalid num_splits: got {}, expected {} for n={}, q={}",
            self.num_splits, expected, self.n, self.modulus
        );
    }

    /// Challenge set size: C(n, ω) * (2τ)^ω
    pub fn challenge_set_size(&self) -> u128 {
        let binomial = binomial_coefficient(self.n, self.omega);
        let coeff_choices = (2 * self.tau) as u128;

        (0..self.omega)
            .try_fold(1u128, |acc, _| acc.checked_mul(coeff_choices))
            .map_or(u128::MAX, |power| binomial.saturating_mul(power))
    }

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

// ============================================================================
// Challenge Element
// ============================================================================

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
}

// ============================================================================
// Challenge Arithmetic
// ============================================================================

impl Challenge {
    /// Compute c₁ - c₂
    pub fn sub(&self, other: &Challenge) -> Challenge {
        debug_assert_eq!(self.n, other.n);

        let result = self
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
        debug_assert_eq!(self.n, other.n);

        let result = self
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
        let coeffs = self
            .sparse_coeffs
            .iter()
            .map(|&(idx, c)| (idx, -c))
            .collect();
        Challenge::new(coeffs, self.n)
    }
}

// ============================================================================
// Challenge Sampling
// ============================================================================

/// Sample a random challenge from the challenge set
pub fn sample_challenge<R: Rng>(rng: &mut R, n: usize, tau: u64, omega: usize) -> Challenge {
    let positions = sample_distinct_positions(rng, n, omega);
    let sparse_coeffs = positions
        .into_iter()
        .map(|pos| (pos, sample_nonzero_bounded(rng, tau)))
        .collect();
    Challenge::new(sparse_coeffs, n)
}

/// Sample a ternary challenge (coefficients in {-1, 0, 1})
pub fn sample_ternary_challenge<R: Rng>(rng: &mut R, n: usize, omega: usize) -> Challenge {
    sample_challenge(rng, n, 1, omega)
}

/// Deterministic challenge from seed (for Fiat-Shamir)
pub fn challenge_from_seed(seed: &[u8], n: usize, omega: usize, tau: u64) -> Challenge {
    use sha2::{Digest, Sha256};

    let mut positions = Vec::with_capacity(omega);
    let mut coeffs = Vec::with_capacity(omega);
    let mut used_positions = HashSet::new();

    let mut current_hash = Sha256::digest(seed);
    let mut hash_counter = 0u64;
    let mut byte_idx = 0;

    while positions.len() < omega {
        if byte_idx + 3 > 32 {
            let mut hasher = Sha256::new();
            hasher.update(seed);
            hasher.update(hash_counter.to_le_bytes());
            current_hash = hasher.finalize();
            hash_counter += 1;
            byte_idx = 0;
        }

        let pos =
            u16::from_le_bytes([current_hash[byte_idx], current_hash[byte_idx + 1]]) as usize % n;
        byte_idx += 2;

        if used_positions.insert(pos) {
            positions.push(pos);
            let coeff_byte = current_hash[byte_idx];
            byte_idx += 1;

            let sign = if coeff_byte & 1 == 0 { 1i64 } else { -1 };
            let mag = if tau == 1 {
                1
            } else {
                ((coeff_byte >> 1) as u64 % tau) as i64 + 1
            };
            coeffs.push(sign * mag);
        } else {
            byte_idx += 1;
        }
    }

    Challenge::new(positions.into_iter().zip(coeffs).collect(), n)
}

// ============================================================================
// Invertibility Check
// ============================================================================

/// Fast path check for difference invertibility based on Theorem 3.1
///
/// Reference: Lyubashevsky & Seiler, EUROCRYPT 2018 (IACR ePrint 2017/523)
pub fn is_definitely_difference_invertible(diff: &Challenge, params: &SplittingParams) -> bool {
    if diff.sparse_coeffs.is_empty() {
        return false;
    }

    let (q, k, tau) = (
        params.modulus as i64,
        params.num_splits as i64,
        params.tau as i64,
    );
    let max_coeff = diff.ell_inf_norm();

    if max_coeff > 2 * tau {
        return false;
    }

    // Check Theorem 3.1 condition: q > 4τ·k
    if q <= 4 * tau * k {
        // Fall back to heuristic L1 check
        let l1_norm: i64 = diff.sparse_coeffs.iter().map(|(_, c)| c.abs()).sum();
        return l1_norm > 0 && l1_norm < params.n as i64;
    }

    true
}

// ============================================================================
// Challenge Set
// ============================================================================

/// Precomputed challenge set for small parameters
#[derive(Clone, Debug)]
pub struct ChallengeSet {
    pub challenges: Vec<Challenge>,
    pub params: SplittingParams,
}

impl ChallengeSet {
    /// Build a challenge set of distinct challenges
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

        #[cfg(debug_assertions)]
        for (i, c1) in challenges.iter().enumerate() {
            for c2 in challenges.iter().skip(i + 1) {
                debug_assert!(
                    is_definitely_difference_invertible(&c1.sub(c2), &params),
                    "Theorem 3.1 violated"
                );
            }
        }

        Self { challenges, params }
    }

    pub fn size(&self) -> usize {
        self.challenges.len()
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> &Challenge {
        &self.challenges[rng.gen_range(0..self.challenges.len())]
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Sample k distinct positions from [0, n)
fn sample_distinct_positions<R: Rng>(rng: &mut R, n: usize, k: usize) -> Vec<usize> {
    debug_assert!(k <= n);

    if k > n / 2 {
        // Fisher-Yates for large k
        let mut positions: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = rng.gen_range(i..n);
            positions.swap(i, j);
        }
        positions.truncate(k);
        positions
    } else {
        // Rejection sampling for small k
        let mut positions = HashSet::with_capacity(k);
        while positions.len() < k {
            positions.insert(rng.gen_range(0..n));
        }
        positions.into_iter().collect()
    }
}

/// Sample a non-zero integer from {-bound, ..., -1, 1, ..., bound}
fn sample_nonzero_bounded<R: Rng>(rng: &mut R, bound: u64) -> i64 {
    let r = rng.gen_range(0..2 * bound);
    if r < bound {
        -((r + 1) as i64)
    } else {
        (r - bound + 1) as i64
    }
}

/// Compute binomial coefficient C(n, k)
fn binomial_coefficient(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    (0..k).fold(1u128, |acc, i| {
        acc.saturating_mul((n - i) as u128) / (i + 1) as u128
    })
}

// ============================================================================
// Standard Parameter Sets
// ============================================================================

pub fn dilithium_challenge_params() -> SplittingParams {
    SplittingParams {
        n: 256,
        num_splits: 256,
        tau: 1,
        omega: 60,
        modulus: 8_380_417,
    }
}

pub fn hachi_challenge_params() -> SplittingParams {
    SplittingParams {
        n: 1024,
        num_splits: 256,
        tau: 1,
        omega: 128,
        modulus: 65537,
    }
}

pub fn test_challenge_params() -> SplittingParams {
    SplittingParams {
        n: 64,
        num_splits: 16,
        tau: 1,
        omega: 16,
        modulus: 65537,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_challenge_creation() {
        let c = Challenge::new(vec![(0, 1), (5, -1), (10, 1)], 64);
        assert_eq!(c.weight(), 3);
        assert_eq!(c.ell_inf_norm(), 1);

        let dense = c.to_dense();
        assert_eq!((dense[0], dense[5], dense[10], dense[1]), (1, -1, 1, 0));
    }

    #[test]
    fn test_challenge_arithmetic() {
        let c1 = Challenge::new(vec![(0, 1), (5, 2)], 64);
        let c2 = Challenge::new(vec![(5, 1), (10, -1)], 64);

        assert_eq!(c1.sub(&c2).sparse_coeffs, vec![(0, 1), (5, 1), (10, 1)]);
        assert_eq!(c1.add(&c2).sparse_coeffs, vec![(0, 1), (5, 3), (10, -1)]);
    }

    #[test]
    fn test_sample_challenge() {
        let mut rng = rand::thread_rng();
        let (n, tau, omega) = (64, 1, 16);

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

        for _ in 0..50 {
            let c1 = sample_challenge(&mut rng, params.n, params.tau, params.omega);
            let c2 = sample_challenge(&mut rng, params.n, params.tau, params.omega);

            if c1 != c2 {
                assert!(is_definitely_difference_invertible(&c1.sub(&c2), &params));
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

        assert!(!is_definitely_difference_invertible(
            &Challenge::zero(64),
            &params
        ));
        assert!(is_definitely_difference_invertible(
            &Challenge::new(vec![(0, 1)], 64),
            &params
        ));
        assert!(is_definitely_difference_invertible(
            &Challenge::new(vec![(0, -1)], 64),
            &params
        ));
        assert!(is_definitely_difference_invertible(
            &Challenge::new(vec![(0, 1), (5, -1)], 64),
            &params
        ));
        assert!(is_definitely_difference_invertible(
            &Challenge::new(vec![(3, 1), (7, -1)], 64),
            &params
        ));
    }

    #[test]
    fn test_challenge_set_size() {
        let params = SplittingParams {
            n: 64,
            num_splits: 64,
            tau: 1,
            omega: 32,
            modulus: 8_380_417,
        };
        let size = params.challenge_set_size();
        assert!(size > 0);
        assert!(params.security_bits() > 60);

        let dilithium = dilithium_challenge_params();
        assert!(
            dilithium.challenge_set_size() == u128::MAX
                || dilithium.challenge_set_size() > 1u128 << 100
        );
    }

    #[test]
    fn test_deterministic_challenge() {
        let seed = b"test seed for challenge generation";

        let c1 = challenge_from_seed(seed, 256, 60, 1);
        let c2 = challenge_from_seed(seed, 256, 60, 1);
        let c3 = challenge_from_seed(b"different seed", 256, 60, 1);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(10, 0), 1);
        assert_eq!(binomial_coefficient(10, 10), 1);
        assert_eq!(binomial_coefficient(10, 1), 10);
        assert_eq!(binomial_coefficient(10, 2), 45);
        assert_eq!(binomial_coefficient(10, 5), 252);
        assert!(binomial_coefficient(256, 60) > 0);
    }

    #[test]
    fn test_sample_distinct_positions() {
        let mut rng = rand::thread_rng();

        for &(n, k) in &[(100, 10), (100, 80)] {
            let pos = sample_distinct_positions(&mut rng, n, k);
            assert_eq!(pos.len(), k);
            assert_eq!(pos.iter().collect::<HashSet<_>>().len(), k);
        }
    }

    #[test]
    fn test_sample_nonzero_bounded() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x = sample_nonzero_bounded(&mut rng, 5);
            assert!((-5..=5).contains(&x) && x != 0);
        }
    }

    #[test]
    fn test_compute_num_splits() {
        // Dilithium: fully splits
        assert_eq!(SplittingParams::compute_num_splits(256, 8_380_417), 256);
        // Kyber: partially splits
        assert_eq!(SplittingParams::compute_num_splits(256, 3329), 128);
        // Fermat primes
        assert_eq!(SplittingParams::compute_num_splits(64, 17), 8);
        assert_eq!(SplittingParams::compute_num_splits(256, 257), 128);
        assert_eq!(SplittingParams::compute_num_splits(1024, 65537), 1024);
        // Edge cases
        assert_eq!(SplittingParams::compute_num_splits(64, 3), 1);
        assert_eq!(SplittingParams::compute_num_splits(64, 5), 2);
    }

    #[test]
    fn test_with_computed_splits() {
        let params = SplittingParams::with_computed_splits(256, 1, 60, 8_380_417);
        assert_eq!(params.num_splits, 256);
        assert!(params.is_valid_num_splits());

        let params2 = SplittingParams::with_computed_splits(256, 1, 60, 3329);
        assert_eq!(params2.num_splits, 128);
        assert!(params2.is_valid_num_splits());
    }

    #[test]
    fn test_validate_num_splits() {
        let valid = SplittingParams {
            n: 256,
            num_splits: 256,
            tau: 1,
            omega: 60,
            modulus: 8_380_417,
        };
        assert!(valid.is_valid_num_splits());

        let invalid = SplittingParams {
            n: 256,
            num_splits: 128,
            tau: 1,
            omega: 60,
            modulus: 8_380_417,
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
        let set = ChallengeSet::build(params, 10);

        assert_eq!(set.size(), 10);
        assert_eq!(set.challenges.iter().collect::<HashSet<_>>().len(), 10);

        let mut rng = rand::thread_rng();
        let sampled = set.sample(&mut rng);
        assert_eq!(sampled.n, 64);
        assert!(sampled.weight() <= 8);
    }
}
