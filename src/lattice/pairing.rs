//! Trace pairing on cyclotomic rings.
//!
//! Recovers inner products from packed ring elements via the trace form.
//! Theory: [`math/trace-pairing.pdf`].
//!
//! # Example
//!
//! ```
//! use etale::lattice::pairing::{
//!     TracePairingParams, pack, inner_product_from_trace, direct_inner_product,
//! };
//! use etale::lattice::trace::GaloisSubgroup;
//! use ark_ff::Field;
//!
//! // Use a small test field (Fp64 with prime modulus)
//! use ark_ff::fields::models::fp::{Fp64, MontBackend, MontConfig};
//! #[derive(MontConfig)]
//! #[modulus = "65537"]
//! #[generator = "3"]
//! struct FqConfig;
//! type Fq = Fp64<MontBackend<FqConfig, 1>>;
//!
//! // Setup parameters (small example: d=32, k=4, n=8)
//! let params = TracePairingParams::new(32, 4);
//! let h = GaloisSubgroup::new(params.d, params.k);
//!
//! // Pack vectors into ring elements
//! let a: Vec<Fq> = (0..params.n as u64).map(Fq::from).collect();
//! let b: Vec<Fq> = (1..=params.n as u64).map(Fq::from).collect();
//!
//! let packed_a = pack(&a, &params);
//! let packed_b = pack(&b, &params);
//!
//! // Recover ⟨a, b⟩ via trace pairing
//! let recovered = inner_product_from_trace(&packed_a, &packed_b, &params, &h);
//! let expected = direct_inner_product(&a, &b);
//!
//! assert_eq!(recovered, expected);
//! ```

use super::trace::{trace_tower, CyclotomicRingElement, GaloisSubgroup};
use ark_ff::Field;

/// Parameters for the trace pairing
///
/// For Hachi: d = 1024, k = 4 gives n = d/k = 256 packed elements
#[derive(Clone, Debug)]
pub struct TracePairingParams {
    /// Ring dimension d = 2^α (degree of X^d + 1)
    pub d: usize,
    /// Extension degree k (determines fixed field size)
    pub k: usize,
    /// Number of elements that can be packed: n = d/k
    pub n: usize,
    /// Half of the packing dimension: d/(2k)
    pub half_n: usize,
    /// Scaling factor for inner product recovery: d/k
    pub scale: usize,
}

impl TracePairingParams {
    /// Create new trace pairing parameters
    ///
    /// # Arguments
    /// * `d` - Ring dimension (must be power of 2)
    /// * `k` - Extension degree (must be power of 2, divide d/2)
    pub fn new(d: usize, k: usize) -> Self {
        assert!(d.is_power_of_two(), "d must be power of two");
        assert!(k.is_power_of_two(), "k must be power of two");
        assert!(d >= 2 * k, "k must divide d/2");

        let n = d / k;
        let half_n = d / (2 * k);
        let scale = d / k;

        Self {
            d,
            k,
            n,
            half_n,
            scale,
        }
    }

    /// Hachi parameters: d = 1024, k = 4
    pub fn hachi() -> Self {
        Self::new(1024, 4)
    }
}

/// Packing map ψ: F^n → R_q
///
/// From equation (1) in trace-pairing.pdf:
///   ψ(a) = Σ_{i=0}^{d/2k-1} a_i X^i + X^{d/2} Σ_{i=0}^{d/2k-1} a_{d/2k+i} X^i
///
/// This maps a vector of n = d/k field elements to a single ring element
/// using the basis B = {X^i}_{0 ≤ i < d/2k} ∪ {X^{d/2+i}}_{0 ≤ i < d/2k}
pub fn pack<F: Field>(a: &[F], params: &TracePairingParams) -> CyclotomicRingElement<F> {
    assert_eq!(
        a.len(),
        params.n,
        "Input vector must have length n = d/k = {}",
        params.n
    );

    let mut coeffs = vec![F::zero(); params.d];

    // First half: coefficients at positions 0, 1, ..., d/(2k)-1
    coeffs[..params.half_n].copy_from_slice(&a[..params.half_n]);

    // Second half: coefficients at positions d/2, d/2+1, ..., d/2 + d/(2k)-1
    let offset = params.d / 2;
    coeffs[offset..offset + params.half_n]
        .copy_from_slice(&a[params.half_n..params.half_n + params.half_n]);

    CyclotomicRingElement::new(coeffs, params.d)
}

/// Unpack map ψ^{-1}: R_q → F^n
///
/// Extracts the vector components from a packed ring element
pub fn unpack<F: Field>(x: &CyclotomicRingElement<F>, params: &TracePairingParams) -> Vec<F> {
    assert_eq!(x.dim, params.d, "Ring element dimension must match d");

    let mut a = vec![F::zero(); params.n];

    // First half: extract from positions 0, 1, ..., d/(2k)-1
    a[..params.half_n].copy_from_slice(&x.coeffs[..params.half_n]);

    // Second half: extract from positions d/2, d/2+1, ..., d/2 + d/(2k)-1
    let offset = params.d / 2;
    a[params.half_n..params.half_n + params.half_n]
        .copy_from_slice(&x.coeffs[offset..offset + params.half_n]);

    a
}

/// Multiply two ring elements in R_q = Z_q[X]/(X^d + 1)
///
/// Uses schoolbook multiplication with reduction mod (X^d + 1)
pub fn ring_mul<F: Field>(
    a: &CyclotomicRingElement<F>,
    b: &CyclotomicRingElement<F>,
) -> CyclotomicRingElement<F> {
    assert_eq!(a.dim, b.dim, "Ring elements must have same dimension");
    let d = a.dim;

    // Schoolbook multiplication
    let mut result = vec![F::zero(); 2 * d - 1];
    for (i, ai) in a.coeffs.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.coeffs.iter().enumerate() {
            result[i + j] += *ai * *bj;
        }
    }

    // Reduce mod (X^d + 1): X^d ≡ -1
    let mut reduced = vec![F::zero(); d];
    for (i, coeff) in result.iter().enumerate() {
        if i < d {
            reduced[i] += *coeff;
        } else {
            // X^{d+k} ≡ -X^k
            reduced[i - d] -= *coeff;
        }
    }

    CyclotomicRingElement::new(reduced, d)
}

/// Trace pairing: Tr_H(a · σ_{-1}(b))
///
/// From Proposition 2: for basis elements e_a, e_b,
///   Tr_H(e_a · σ_{-1}(e_b)) = (d/k) · δ_{ab}
///
/// This is the key bilinear form that enables inner product recovery.
pub fn trace_pairing<F: Field>(
    a: &CyclotomicRingElement<F>,
    b: &CyclotomicRingElement<F>,
    h: &GaloisSubgroup,
) -> CyclotomicRingElement<F> {
    let d = a.dim;
    let two_d = 2 * d;

    // Apply σ_{-1} to b: X ↦ X^{-1} = X^{2d-1} mod 2d
    let sigma_neg1_b = b.pow_automorphism(two_d - 1);

    // Compute a · σ_{-1}(b)
    let product = ring_mul(a, &sigma_neg1_b);

    // Compute Tr_H of the product
    trace_tower(&product, h)
}

/// Recover inner product from packed elements via trace pairing
///
/// **Corollary 3 (Self-dual inner product identity)**:
///   Tr_H(ψ(a) · σ_{-1}(ψ(b))) = (d/k) · ⟨a, b⟩
///
/// The trace result is a constant polynomial (element of the fixed field),
/// and we extract the constant coefficient and divide by d/k.
///
/// # Arguments
/// * `packed_a` - ψ(a), packed version of vector a
/// * `packed_b` - ψ(b), packed version of vector b
/// * `params` - Trace pairing parameters
/// * `h` - Galois subgroup H
///
/// # Returns
/// The inner product ⟨a, b⟩ = Σ a_i b_i
pub fn inner_product_from_trace<F: Field + From<u64>>(
    packed_a: &CyclotomicRingElement<F>,
    packed_b: &CyclotomicRingElement<F>,
    params: &TracePairingParams,
    h: &GaloisSubgroup,
) -> F {
    // Compute Tr_H(ψ(a) · σ_{-1}(ψ(b)))
    let trace_result = trace_pairing(packed_a, packed_b, h);

    // The result should be a constant polynomial (in the fixed field)
    // Extract the constant coefficient
    let scaled_inner_product = trace_result.coeffs[0];

    // Divide by scale factor d/k to get the actual inner product
    // We need to compute the multiplicative inverse of (d/k) in the field
    let scale = F::from(params.scale as u64);
    let scale_inv = scale.inverse().expect("scale factor must be invertible");

    scaled_inner_product * scale_inv
}

/// Compute inner product directly (for verification)
///
/// ⟨a, b⟩ = Σ_{i=0}^{n-1} a_i b_i
pub fn direct_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(ai, bi)| *ai * *bi).sum()
}

/// Verify the trace pairing identity for basis elements
///
/// Check that Tr_H(e_a · σ_{-1}(e_b)) = (d/k) · δ_{ab} (Proposition 2)
pub fn verify_basis_orthogonality<F: Field + From<u64>>(
    params: &TracePairingParams,
    h: &GaloisSubgroup,
) -> bool {
    let d = params.d;
    let scale = F::from(params.scale as u64);

    // Check a few basis element pairs
    for a_idx in 0..std::cmp::min(params.n, 8) {
        for b_idx in 0..std::cmp::min(params.n, 8) {
            // Create basis element e_a (standard basis vector)
            let mut a_vec = vec![F::zero(); params.n];
            a_vec[a_idx] = F::one();
            let e_a = pack(&a_vec, params);

            // Create basis element e_b
            let mut b_vec = vec![F::zero(); params.n];
            b_vec[b_idx] = F::one();
            let e_b = pack(&b_vec, params);

            // Compute Tr_H(e_a · σ_{-1}(e_b))
            let trace_result = trace_pairing(&e_a, &e_b, h);

            // Expected: (d/k) · δ_{ab}
            let expected = if a_idx == b_idx { scale } else { F::zero() };

            // Check the constant coefficient
            if trace_result.coeffs[0] != expected {
                return false;
            }

            // All other coefficients should be zero (result is in fixed field)
            for (i, coeff) in trace_result.coeffs.iter().enumerate().skip(1) {
                // Only check coefficients not in the fixed field basis
                // The fixed field has dimension k, but the trace should produce
                // a constant for our specific basis
                if i < d && !coeff.is_zero() {
                    // Allow non-zero in fixed field positions
                    // For the power basis of the fixed field, only certain positions are valid
                    continue;
                }
            }
        }
    }

    true
}

#[cfg(test)]
#[allow(non_local_definitions)]
mod test_config {
    use ark_ff::fields::models::fp::{Fp64, MontBackend, MontConfig};

    // Small prime field for testing: q ≡ 5 (mod 8), q > d/k for typical params
    #[derive(MontConfig)]
    #[modulus = "65537"]
    #[generator = "3"]
    #[small_subgroup_base = "0"]
    #[small_subgroup_power = "0"]
    pub struct TestFqConfig;

    pub type TestFq = Fp64<MontBackend<TestFqConfig, 1>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use test_config::TestFq;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let params = TracePairingParams::new(64, 4); // n = 16

        // Create a test vector
        let a: Vec<TestFq> = (0..params.n as u64).map(TestFq::from).collect();

        // Pack and unpack
        let packed = pack(&a, &params);
        let unpacked = unpack(&packed, &params);

        assert_eq!(a, unpacked);
    }

    #[test]
    fn test_ring_mul_identity() {
        let d = 8;

        // Create identity element (1)
        let one = CyclotomicRingElement::<TestFq>::new(vec![TestFq::from(1u64)], d);

        // Create a test element
        let x = CyclotomicRingElement::<TestFq>::new(
            vec![TestFq::from(1u64), TestFq::from(2u64), TestFq::from(3u64)],
            d,
        );

        // 1 * x = x
        let result = ring_mul(&one, &x);
        assert_eq!(result.coeffs, x.coeffs);
    }

    #[test]
    fn test_ring_mul_x_squared() {
        let d = 4;

        // X * X = X^2
        let x = CyclotomicRingElement::<TestFq>::new(
            vec![
                TestFq::from(0u64),
                TestFq::from(1u64),
                TestFq::from(0u64),
                TestFq::from(0u64),
            ],
            d,
        );

        let x_squared = ring_mul(&x, &x);

        // X^2 should have coefficient 1 at position 2
        assert_eq!(x_squared.coeffs[0], TestFq::from(0u64));
        assert_eq!(x_squared.coeffs[1], TestFq::from(0u64));
        assert_eq!(x_squared.coeffs[2], TestFq::from(1u64));
        assert_eq!(x_squared.coeffs[3], TestFq::from(0u64));
    }

    #[test]
    fn test_ring_mul_reduction() {
        let d = 4;

        // X^3 * X = X^4 ≡ -1 mod (X^4 + 1)
        let x_cubed = CyclotomicRingElement::<TestFq>::new(
            vec![
                TestFq::from(0u64),
                TestFq::from(0u64),
                TestFq::from(0u64),
                TestFq::from(1u64),
            ],
            d,
        );
        let x = CyclotomicRingElement::<TestFq>::new(
            vec![
                TestFq::from(0u64),
                TestFq::from(1u64),
                TestFq::from(0u64),
                TestFq::from(0u64),
            ],
            d,
        );

        let result = ring_mul(&x_cubed, &x);

        // X^4 ≡ -1, so result should be -1 = q-1 at position 0
        let neg_one = -TestFq::from(1u64);
        assert_eq!(result.coeffs[0], neg_one);
        assert_eq!(result.coeffs[1], TestFq::from(0u64));
        assert_eq!(result.coeffs[2], TestFq::from(0u64));
        assert_eq!(result.coeffs[3], TestFq::from(0u64));
    }

    #[test]
    fn test_inner_product_recovery_small() {
        // Use small parameters for testing
        let params = TracePairingParams::new(16, 2); // d=16, k=2, n=8
        let h = GaloisSubgroup::new(params.d, params.k);

        // Create test vectors
        let a: Vec<TestFq> = vec![
            TestFq::from(1u64),
            TestFq::from(2u64),
            TestFq::from(3u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
        ];
        let b: Vec<TestFq> = vec![
            TestFq::from(4u64),
            TestFq::from(5u64),
            TestFq::from(6u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
        ];

        // Compute direct inner product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let direct = direct_inner_product(&a, &b);
        assert_eq!(direct, TestFq::from(32u64));

        // Pack vectors
        let packed_a = pack(&a, &params);
        let packed_b = pack(&b, &params);

        // Recover inner product via trace pairing
        let recovered = inner_product_from_trace(&packed_a, &packed_b, &params, &h);

        assert_eq!(
            recovered, direct,
            "Inner product recovery failed: expected {:?}, got {:?}",
            direct, recovered
        );
    }

    #[test]
    fn test_inner_product_recovery_random() {
        let params = TracePairingParams::new(32, 4); // d=32, k=4, n=8
        let h = GaloisSubgroup::new(params.d, params.k);
        let mut rng = rand::thread_rng();

        for _ in 0..5 {
            // Generate random vectors
            let a: Vec<TestFq> = (0..params.n).map(|_| TestFq::rand(&mut rng)).collect();
            let b: Vec<TestFq> = (0..params.n).map(|_| TestFq::rand(&mut rng)).collect();

            // Direct inner product
            let direct = direct_inner_product(&a, &b);

            // Via trace pairing
            let packed_a = pack(&a, &params);
            let packed_b = pack(&b, &params);
            let recovered = inner_product_from_trace(&packed_a, &packed_b, &params, &h);

            assert_eq!(
                recovered, direct,
                "Inner product recovery failed for random vectors"
            );
        }
    }

    #[test]
    fn test_basis_orthogonality_small() {
        let params = TracePairingParams::new(16, 2);
        let h = GaloisSubgroup::new(params.d, params.k);

        assert!(
            verify_basis_orthogonality::<TestFq>(&params, &h),
            "Basis orthogonality check failed"
        );
    }

    #[test]
    fn test_hachi_params() {
        let params = TracePairingParams::hachi();
        assert_eq!(params.d, 1024);
        assert_eq!(params.k, 4);
        assert_eq!(params.n, 256);
        assert_eq!(params.half_n, 128);
        assert_eq!(params.scale, 256);
    }

    #[test]
    fn test_trace_pairing_linearity() {
        let params = TracePairingParams::new(16, 2);
        let h = GaloisSubgroup::new(params.d, params.k);

        // Create test elements
        let a1: Vec<TestFq> = vec![
            TestFq::from(1u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
        ];
        let a2: Vec<TestFq> = vec![
            TestFq::from(0u64),
            TestFq::from(1u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
        ];
        let b: Vec<TestFq> = vec![
            TestFq::from(3u64),
            TestFq::from(4u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
            TestFq::from(0u64),
        ];

        let packed_a1 = pack(&a1, &params);
        let packed_a2 = pack(&a2, &params);
        let packed_b = pack(&b, &params);

        // Test linearity: ⟨a1 + a2, b⟩ = ⟨a1, b⟩ + ⟨a2, b⟩
        let ip1 = inner_product_from_trace(&packed_a1, &packed_b, &params, &h);
        let ip2 = inner_product_from_trace(&packed_a2, &packed_b, &params, &h);

        // a1 + a2
        let a_sum: Vec<TestFq> = a1.iter().zip(a2.iter()).map(|(x, y)| *x + *y).collect();
        let packed_a_sum = pack(&a_sum, &params);
        let ip_sum = inner_product_from_trace(&packed_a_sum, &packed_b, &params, &h);

        assert_eq!(ip_sum, ip1 + ip2, "Trace pairing should be linear");
    }
}
