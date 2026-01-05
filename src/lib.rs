//! # Fiat-Shamir with Aborts
//!
//! Implementation of the rejection sampling framework from Lyubashevsky's 2009 paper
//! "Fiat-Shamir with Aborts: Applications to Lattice and Factoring-Based Signatures"
//!
//! ## Overview
//!
//! The key insight is that in lattice-based identification/signature schemes, the prover's
//! response z = y + sc (where y is random, s is the secret, c is the challenge) leaks
//! information about s. Rejection sampling ensures that the distribution of z is
//! independent of s by only outputting z when it falls in a "safe" range.
//!
//! ## The Protocol (simplified)
//!
//! 1. Prover samples y uniformly from [-B, B]^n
//! 2. Prover sends commitment w = Ay mod q
//! 3. Verifier sends challenge c (small)
//! 4. Prover computes z = y + sc
//! 5. **Rejection sampling**: If ||z||_∞ > B - ||sc||_∞, ABORT and restart
//! 6. Otherwise, output z
//!
//! The abort ensures that z is uniformly distributed over a range independent of s.

use rand::Rng;
use sha2::{Digest, Sha256};
use std::fmt;

/// Parameters for the scheme
#[derive(Clone, Debug)]
pub struct IdentificationSchemeParameters {
    /// Dimension of vectors
    pub n: usize,
    /// Modulus for the ring Z_q
    pub q: i64,
    /// Bound for the commitment randomness y: sampled from [-b, b]
    pub b: i64,
    /// Bound for the challenge coefficients
    pub challenge_bound: i64,
    /// Bound for the secret key coefficients
    pub secret_bound: i64,
}

impl Default for IdentificationSchemeParameters {
    fn default() -> Self {
        // Small toy parameters for demonstration
        // Real parameters would be much larger (n ~ 512-1024, q ~ 2^23)
        IdentificationSchemeParameters {
            n: 4,
            q: 7681, // A small prime used in some lattice schemes
            b: 1000,
            challenge_bound: 1,
            secret_bound: 1,
        }
    }
}

impl IdentificationSchemeParameters {
    /// Compute the rejection threshold
    /// z is accepted if ||z||_∞ <= b - challenge_bound * secret_bound * n
    pub fn rejection_bound(&self) -> i64 {
        // Conservative bound: account for worst-case ||sc||_∞
        self.b - self.challenge_bound * self.secret_bound * (self.n as i64)
    }
}

/// A vector in Z_q^n
#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    pub coeffs: Vec<i64>,
    pub modulus: i64,
}

impl Vector {
    pub fn new(coeffs: Vec<i64>, modulus: i64) -> Self {
        let reduced: Vec<i64> = coeffs
            .into_iter()
            .map(|c| ((c % modulus) + modulus) % modulus)
            .collect();
        Vector {
            coeffs: reduced,
            modulus,
        }
    }

    pub fn zero(n: usize, modulus: i64) -> Self {
        Vector {
            coeffs: vec![0; n],
            modulus,
        }
    }

    pub fn random<R: Rng>(rng: &mut R, n: usize, bound: i64, modulus: i64) -> Self {
        let coeffs: Vec<i64> = (0..n).map(|_| rng.gen_range(-bound..=bound)).collect();
        Vector::new(coeffs, modulus)
    }

    pub fn random_ternary<R: Rng>(rng: &mut R, n: usize, modulus: i64) -> Self {
        // Ternary vector: coefficients in {-1, 0, 1}
        let coeffs: Vec<i64> = (0..n).map(|_| rng.gen_range(-1..=1)).collect();
        Vector::new(coeffs, modulus)
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// ell-infinity norm (maximum absolute value of centered representatives)
    pub fn ell_inf_norm(&self) -> i64 {
        self.coeffs
            .iter()
            .map(|&c| {
                // Center the coefficient to [-q/2, q/2]
                let centered = if c > self.modulus / 2 {
                    c - self.modulus
                } else {
                    c
                };
                centered.abs()
            })
            .max()
            .unwrap_or(0)
    }

    /// Centered representative (coefficients in [-q/2, q/2])
    pub fn centered(&self) -> Vec<i64> {
        self.coeffs
            .iter()
            .map(|&c| {
                if c > self.modulus / 2 {
                    c - self.modulus
                } else {
                    c
                }
            })
            .collect()
    }

    /// Add two vectors
    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.coeffs.len(), other.coeffs.len());
        assert_eq!(self.modulus, other.modulus);
        let coeffs: Vec<i64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| (a + b) % self.modulus)
            .collect();
        Vector::new(coeffs, self.modulus)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: i64) -> Vector {
        let coeffs: Vec<i64> = self.coeffs.iter().map(|&c| c * scalar).collect();
        Vector::new(coeffs, self.modulus)
    }

    /// Inner product
    pub fn inner_product(&self, other: &Vector) -> i64 {
        assert_eq!(self.coeffs.len(), other.coeffs.len());
        let sum: i64 = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        ((sum % self.modulus) + self.modulus) % self.modulus
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (mod {})", self.centered(), self.modulus)
    }
}

/// A matrix in Z_q^{m x n}
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: Vec<Vector>,
    pub modulus: i64,
}

impl Matrix {
    pub fn new(rows: Vec<Vector>) -> Self {
        let modulus = rows.first().map(|r| r.modulus).unwrap_or(1);
        Matrix { rows, modulus }
    }

    pub fn random<R: Rng>(rng: &mut R, m: usize, n: usize, modulus: i64) -> Self {
        let rows: Vec<Vector> = (0..m)
            .map(|_| {
                let coeffs: Vec<i64> = (0..n).map(|_| rng.gen_range(0..modulus)).collect();
                Vector::new(coeffs, modulus)
            })
            .collect();
        Matrix::new(rows)
    }

    /// Matrix-vector product Av
    pub fn mul_vec(&self, v: &Vector) -> Vector {
        let coeffs: Vec<i64> = self.rows.iter().map(|row| row.inner_product(v)).collect();
        Vector::new(coeffs, self.modulus)
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn num_cols(&self) -> usize {
        self.rows.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// Secret key: a short vector s
#[derive(Clone, Debug)]
pub struct SecretKey {
    pub s: Vector,
}

/// Public key: matrix A and t = As mod q
#[derive(Clone, Debug)]
pub struct PublicKey {
    pub a: Matrix,
    pub t: Vector,
}

/// Key pair
#[derive(Clone, Debug)]
pub struct KeyPair {
    pub sk: SecretKey,
    pub pk: PublicKey,
}

/// Commitment (first message from prover)
#[derive(Clone, Debug)]
pub struct Commitment {
    pub w: Vector,
}

/// Prover's state (kept secret during protocol)
#[derive(Clone, Debug)]
pub struct ProverState {
    pub y: Vector,
}

/// Challenge (from verifier, or hash in Fiat-Shamir)
#[derive(Clone, Debug)]
pub struct Challenge {
    pub c: i64,
}

/// Response (final message from prover, or abort)
#[derive(Clone, Debug)]
pub enum Response {
    Valid(Vector),
    Abort,
}

/// Statistics for rejection sampling
#[derive(Clone, Debug, Default)]
pub struct RejectionSamplingStats {
    pub total_attempts: u64,
    pub aborts: u64,
    pub successes: u64,
}

impl RejectionSamplingStats {
    pub fn abort_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            0.0
        } else {
            self.aborts as f64 / self.total_attempts as f64
        }
    }
}

/// The identification scheme with rejection sampling
pub struct IdentificationScheme {
    pub params: IdentificationSchemeParameters,
    pub stats: RejectionSamplingStats,
}

impl IdentificationScheme {
    pub fn new(params: IdentificationSchemeParameters) -> Self {
        IdentificationScheme {
            params,
            stats: RejectionSamplingStats::default(),
        }
    }

    /// Generate a key pair
    pub fn keygen<R: Rng>(&self, rng: &mut R) -> KeyPair {
        let n = self.params.n;
        let q = self.params.q;

        // Secret key: short vector
        let s = Vector::random_ternary(rng, n, q);

        // Public matrix A (random)
        let a = Matrix::random(rng, n, n, q);

        // Public key component t = As
        let t = a.mul_vec(&s);

        KeyPair {
            sk: SecretKey { s },
            pk: PublicKey { a, t },
        }
    }

    /// Prover's first message: commitment
    pub fn prover_commit<R: Rng>(&self, rng: &mut R, pk: &PublicKey) -> (Commitment, ProverState) {
        let n = self.params.n;
        let q = self.params.q;
        let b = self.params.b;

        // Sample y uniformly from [-b, b]^n
        let y = Vector::random(rng, n, b, q);

        // Commitment w = Ay mod q
        let w = pk.a.mul_vec(&y);

        (Commitment { w }, ProverState { y })
    }

    /// Generate random challenge,
    pub fn random_challenge<R: Rng>(&self, rng: &mut R) -> Challenge {
        let c = rng.gen_range(-self.params.challenge_bound..=self.params.challenge_bound);
        Challenge { c }
    }

    /// Hash-based challenge for Fiat-Shamir transform
    pub fn hash_challenge(&self, w: &Vector, message: &[u8]) -> Challenge {
        let mut hasher = Sha256::new();

        // Hash the commitment
        for coeff in &w.coeffs {
            hasher.update(coeff.to_le_bytes());
        }

        // Hash the message
        hasher.update(message);

        let result = hasher.finalize();

        // Extract challenge from hash (simplified: just use first bytes)
        let hash_value = i64::from_le_bytes(result[0..8].try_into().unwrap());
        let c = hash_value.rem_euclid(2 * self.params.challenge_bound + 1)
            - self.params.challenge_bound;

        Challenge { c }
    }

    /// Prover's response with rejection sampling
    /// This is the KEY part: we abort if z would leak information about s
    pub fn prover_respond(
        &mut self,
        sk: &SecretKey,
        state: &ProverState,
        challenge: &Challenge,
    ) -> Response {
        self.stats.total_attempts += 1;

        // Compute z = y + sc
        let sc = sk.s.scalar_mul(challenge.c);
        let z = state.y.add(&sc);

        // REJECTION SAMPLING: Check if z is in the safe range
        // We need ||z||_∞ <= B - ||sc||_∞ to ensure z's distribution is independent of s
        let rejection_bound = self.params.rejection_bound();
        let z_norm = z.ell_inf_norm();

        if z_norm > rejection_bound {
            // ABORT: z would leak information about s
            self.stats.aborts += 1;
            Response::Abort
        } else {
            // Accept: z is in the safe range
            self.stats.successes += 1;
            Response::Valid(z)
        }
    }

    /// Verifier checks the response
    pub fn verify(
        &self,
        pk: &PublicKey,
        commitment: &Commitment,
        challenge: &Challenge,
        z: &Vector,
    ) -> bool {
        // Check 1: z has small norm
        let rejection_bound = self.params.rejection_bound();
        if z.ell_inf_norm() > rejection_bound {
            return false;
        }

        // Check 2: Az = w + tc (mod q)
        // Equivalently: Az - tc = w (mod q)
        let az = pk.a.mul_vec(z);
        let tc = pk.t.scalar_mul(challenge.c);
        let lhs = az;
        let rhs = commitment.w.add(&tc);

        lhs == rhs
    }

    /// Run the full identification protocol (with retries on abort)
    /// Generates a proof that the prover knows the secret key
    pub fn prove<R: Rng>(
        &mut self,
        rng: &mut R,
        keypair: &KeyPair,
        max_attempts: usize,
    ) -> Option<(Commitment, Challenge, Vector)> {
        for _ in 0..max_attempts {
            // Step 1: Prover commits
            let (commitment, state) = self.prover_commit(rng, &keypair.pk);

            // Step 2: Verifier challenges
            let challenge = self.random_challenge(rng);

            // Step 3: Prover responds (may abort)
            match self.prover_respond(&keypair.sk, &state, &challenge) {
                Response::Valid(z) => {
                    return Some((commitment, challenge, z));
                }
                Response::Abort => {
                    // Try again
                    continue;
                }
            }
        }
        None
    }
}

/// Signature scheme using Fiat-Shamir transform
pub struct SignatureScheme {
    pub id_scheme: IdentificationScheme,
}

/// A signature
#[derive(Clone, Debug)]
pub struct Signature {
    pub w: Vector,
    pub z: Vector,
}

impl SignatureScheme {
    pub fn new(params: IdentificationSchemeParameters) -> Self {
        SignatureScheme {
            id_scheme: IdentificationScheme::new(params),
        }
    }

    pub fn keygen<R: Rng>(&self, rng: &mut R) -> KeyPair {
        self.id_scheme.keygen(rng)
    }

    /// Sign a message (with retries on abort)
    pub fn sign<R: Rng>(
        &mut self,
        rng: &mut R,
        keypair: &KeyPair,
        message: &[u8],
        max_attempts: usize,
    ) -> Option<Signature> {
        for _ in 0..max_attempts {
            // Step 1: Prover commits
            let (commitment, state) = self.id_scheme.prover_commit(rng, &keypair.pk);

            // Step 2: Challenge is hash of commitment and message (Fiat-Shamir)
            let challenge = self.id_scheme.hash_challenge(&commitment.w, message);

            // Step 3: Prover responds (may abort)
            match self
                .id_scheme
                .prover_respond(&keypair.sk, &state, &challenge)
            {
                Response::Valid(z) => {
                    return Some(Signature { w: commitment.w, z });
                }
                Response::Abort => {
                    continue;
                }
            }
        }
        None
    }

    /// Verify a signature
    pub fn verify(&self, pk: &PublicKey, message: &[u8], sig: &Signature) -> bool {
        // Recompute challenge from commitment and message
        let challenge = self.id_scheme.hash_challenge(&sig.w, message);

        // Use identification scheme verification
        let commitment = Commitment { w: sig.w.clone() };
        self.id_scheme.verify(pk, &commitment, &challenge, &sig.z)
    }

    pub fn stats(&self) -> &RejectionSamplingStats {
        &self.id_scheme.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn get_rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(12345)
    }

    #[test]
    fn test_vector_operations() {
        let q = 7681;
        let v1 = Vector::new(vec![1, 2, 3, 4], q);
        let v2 = Vector::new(vec![5, 6, 7, 8], q);

        let sum = v1.add(&v2);
        assert_eq!(sum.coeffs, vec![6, 8, 10, 12]);

        let scaled = v1.scalar_mul(2);
        assert_eq!(scaled.coeffs, vec![2, 4, 6, 8]);

        let inner = v1.inner_product(&v2);
        assert_eq!(inner, 5 + 12 + 21 + 32); // 70
    }

    #[test]
    fn test_vector_linf_norm() {
        let q = 101;
        // Coefficients that wrap around
        let v = Vector::new(vec![100, 50, 1], q); // 100 mod 101 = -1 centered
        assert_eq!(v.ell_inf_norm(), 50);

        let v2 = Vector::new(vec![-10, 20, -30], q);
        assert_eq!(v2.ell_inf_norm(), 30);
    }

    #[test]
    fn test_matrix_vector_mul() {
        let q = 101;
        let row1 = Vector::new(vec![1, 0, 0], q);
        let row2 = Vector::new(vec![0, 1, 0], q);
        let row3 = Vector::new(vec![0, 0, 1], q);
        let identity = Matrix::new(vec![row1, row2, row3]);

        let v = Vector::new(vec![5, 10, 15], q);
        let result = identity.mul_vec(&v);
        assert_eq!(result.coeffs, vec![5, 10, 15]);
    }

    #[test]
    fn test_keygen() {
        let mut rng = get_rng();
        let params = IdentificationSchemeParameters::default();
        let scheme = IdentificationScheme::new(params.clone());

        let keypair = scheme.keygen(&mut rng);

        // Check that t = As
        let computed_t = keypair.pk.a.mul_vec(&keypair.sk.s);
        assert_eq!(keypair.pk.t, computed_t);

        // Check secret key has small coefficients
        assert!(keypair.sk.s.ell_inf_norm() <= params.secret_bound);
    }

    #[test]
    fn test_identification_honest() {
        let mut rng = get_rng();
        let params = IdentificationSchemeParameters::default();
        let mut scheme = IdentificationScheme::new(params);

        let keypair = scheme.keygen(&mut rng);

        // Run identification (may need multiple attempts due to aborts)
        let result = scheme.prove(&mut rng, &keypair, 100);
        assert!(result.is_some(), "Identification should succeed");

        let (commitment, challenge, z) = result.unwrap();

        // Verify should pass
        assert!(scheme.verify(&keypair.pk, &commitment, &challenge, &z));

        println!("Identification stats: {:?}", scheme.stats);
        println!("Abort rate: {:.2}%", scheme.stats.abort_rate() * 100.0);
    }

    #[test]
    fn test_identification_fails_wrong_key() {
        let mut rng = get_rng();
        let params = IdentificationSchemeParameters::default();
        let mut scheme = IdentificationScheme::new(params);

        let keypair = scheme.keygen(&mut rng);
        let wrong_keypair = scheme.keygen(&mut rng);

        // Generate valid proof with correct key
        let result = scheme.prove(&mut rng, &keypair, 100);
        assert!(result.is_some());

        let (commitment, challenge, z) = result.unwrap();

        // Verify with wrong public key should fail
        assert!(!scheme.verify(&wrong_keypair.pk, &commitment, &challenge, &z));
    }

    #[test]
    fn test_rejection_sampling_distribution() {
        // Test that rejection sampling produces roughly uniform distribution
        let mut rng = get_rng();
        let params = IdentificationSchemeParameters {
            n: 2,
            q: 101,
            b: 50,
            challenge_bound: 1,
            secret_bound: 1,
        };
        let mut scheme = IdentificationScheme::new(params.clone());
        let keypair = scheme.keygen(&mut rng);

        // Collect many successful responses
        let mut z_samples: Vec<Vector> = Vec::new();
        let num_samples = 1000;

        while z_samples.len() < num_samples {
            if let Some((_, _, z)) = scheme.prove(&mut rng, &keypair, 10) {
                z_samples.push(z);
            }
        }

        // Check that z values are within the rejection bound
        let rejection_bound = params.rejection_bound();
        for z in &z_samples {
            assert!(
                z.ell_inf_norm() <= rejection_bound,
                "z should be within rejection bound"
            );
        }

        // Basic uniformity check: mean should be close to 0
        let sum: i64 = z_samples.iter().flat_map(|z| z.centered()).sum();
        let mean = sum as f64 / (z_samples.len() * params.n) as f64;
        assert!(mean.abs() < 5.0, "Mean should be close to 0, got {}", mean);

        println!("Collected {} samples", z_samples.len());
        println!("Mean coefficient: {:.2}", mean);
        println!("Abort rate: {:.2}%", scheme.stats.abort_rate() * 100.0);
    }

    #[test]
    fn test_signature_scheme() {
        let mut rng = get_rng();
        let params = IdentificationSchemeParameters::default();
        let mut scheme = SignatureScheme::new(params);

        let keypair = scheme.keygen(&mut rng);
        let message = b"Hello, lattice cryptography!";

        // Sign
        let sig = scheme.sign(&mut rng, &keypair, message, 100);
        assert!(sig.is_some(), "Signing should succeed");

        let sig = sig.unwrap();

        // Verify
        assert!(scheme.verify(&keypair.pk, message, &sig));

        // Wrong message should fail
        let wrong_message = b"Wrong message";
        assert!(!scheme.verify(&keypair.pk, wrong_message, &sig));

        println!("Signature stats: {:?}", scheme.stats());
    }

    #[test]
    fn test_signature_wrong_key() {
        let mut rng = get_rng();
        let params = IdentificationSchemeParameters::default();
        let mut scheme = SignatureScheme::new(params);

        let keypair = scheme.keygen(&mut rng);
        let wrong_keypair = scheme.keygen(&mut rng);
        let message = b"Test message";

        let sig = scheme.sign(&mut rng, &keypair, message, 100).unwrap();

        // Verify with wrong public key should fail
        assert!(!scheme.verify(&wrong_keypair.pk, message, &sig));
    }

    #[test]
    fn test_abort_rate_varies_with_params() {
        let mut rng = get_rng();

        // Tight parameters (high abort rate)
        let tight_params = IdentificationSchemeParameters {
            n: 4,
            q: 7681,
            b: 100,
            challenge_bound: 1,
            secret_bound: 1,
        };

        // Loose parameters (low abort rate)
        let loose_params = IdentificationSchemeParameters {
            n: 4,
            q: 7681,
            b: 1000,
            challenge_bound: 1,
            secret_bound: 1,
        };

        let mut tight_scheme = IdentificationScheme::new(tight_params);
        let mut loose_scheme = IdentificationScheme::new(loose_params);

        let tight_kp = tight_scheme.keygen(&mut rng);
        let loose_kp = loose_scheme.keygen(&mut rng);

        // Run many identifications
        for _ in 0..100 {
            tight_scheme.prove(&mut rng, &tight_kp, 100);
            loose_scheme.prove(&mut rng, &loose_kp, 100);
        }

        println!(
            "Tight params abort rate: {:.2}%",
            tight_scheme.stats.abort_rate() * 100.0
        );
        println!(
            "Loose params abort rate: {:.2}%",
            loose_scheme.stats.abort_rate() * 100.0
        );

        // Tight params should have higher abort rate
        assert!(
            tight_scheme.stats.abort_rate() > loose_scheme.stats.abort_rate(),
            "Tighter parameters should have higher abort rate"
        );
    }

    #[test]
    fn test_deterministic_challenge() {
        let params = IdentificationSchemeParameters::default();
        let scheme = IdentificationScheme::new(params.clone());

        let w = Vector::new(vec![1, 2, 3, 4], params.q);
        let message = b"test message";

        let c1 = scheme.hash_challenge(&w, message);
        let c2 = scheme.hash_challenge(&w, message);

        assert_eq!(c1.c, c2.c, "Same inputs should produce same challenge");

        let different_message = b"different message";
        let c3 = scheme.hash_challenge(&w, different_message);

        // Different messages should (almost certainly) produce different challenges
        // This could theoretically fail due to collision, but extremely unlikely
        assert_ne!(
            c1.c, c3.c,
            "Different messages should produce different challenges"
        );
    }

    #[test]
    fn test_zero_knowledge_property() {
        // Informal test: the distribution of z should look the same
        // regardless of the secret key
        let mut rng = get_rng();
        let params = IdentificationSchemeParameters {
            n: 2,
            q: 7681,
            b: 500,
            challenge_bound: 1,
            secret_bound: 1,
        };

        let mut scheme1 = IdentificationScheme::new(params.clone());
        let mut scheme2 = IdentificationScheme::new(params.clone());

        // Two different keys
        let kp1 = scheme1.keygen(&mut rng);
        let kp2 = scheme2.keygen(&mut rng);

        // Collect z samples from both
        let mut z1_samples: Vec<i64> = Vec::new();
        let mut z2_samples: Vec<i64> = Vec::new();

        for _ in 0..500 {
            if let Some((_, _, z)) = scheme1.prove(&mut rng, &kp1, 10) {
                z1_samples.extend(z.centered());
            }
            if let Some((_, _, z)) = scheme2.prove(&mut rng, &kp2, 10) {
                z2_samples.extend(z.centered());
            }
        }

        // Compare distributions (very rough check)
        let mean1: f64 = z1_samples.iter().sum::<i64>() as f64 / z1_samples.len() as f64;
        let mean2: f64 = z2_samples.iter().sum::<i64>() as f64 / z2_samples.len() as f64;

        let var1: f64 = z1_samples
            .iter()
            .map(|&x| (x as f64 - mean1).powi(2))
            .sum::<f64>()
            / z1_samples.len() as f64;
        let var2: f64 = z2_samples
            .iter()
            .map(|&x| (x as f64 - mean2).powi(2))
            .sum::<f64>()
            / z2_samples.len() as f64;

        println!("Key 1 - Mean: {:.2}, Variance: {:.2}", mean1, var1);
        println!("Key 2 - Mean: {:.2}, Variance: {:.2}", mean2, var2);

        // The means and variances should be similar (both close to uniform)
        assert!(
            (mean1 - mean2).abs() < 10.0,
            "Means should be similar regardless of key"
        );
        assert!(
            (var1 - var2).abs() / var1.max(var2) < 0.2,
            "Variances should be similar regardless of key"
        );
    }
}
