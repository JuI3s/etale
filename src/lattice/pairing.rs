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

// ============================================================================
// Trace Pairing Parameters
// ============================================================================

/// Parameters for the trace pairing
///
/// For Hachi: d = 1024, k = 4 gives n = d/k = 256 packed elements
#[derive(Clone, Debug)]
pub struct TracePairingParams {
    /// Ring dimension d = 2^α
    pub d: usize,
    /// Extension degree k
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
    pub fn new(d: usize, k: usize) -> Self {
        debug_assert!(d.is_power_of_two(), "d must be power of two");
        debug_assert!(k.is_power_of_two(), "k must be power of two");
        debug_assert!(d >= 2 * k, "k must divide d/2");

        let n = d / k;
        Self {
            d,
            k,
            n,
            half_n: d / (2 * k),
            scale: n,
        }
    }

    /// Hachi parameters: d = 1024, k = 4
    pub fn hachi() -> Self {
        Self::new(1024, 4)
    }
}

// ============================================================================
// Packing / Unpacking
// ============================================================================

/// Packing map ψ: F^n → R_q
///
/// From equation (1) in trace-pairing.pdf:
///   ψ(a) = Σ_{i=0}^{d/2k-1} a_i X^i + X^{d/2} Σ_{i=0}^{d/2k-1} a_{d/2k+i} X^i
pub fn pack<F: Field>(a: &[F], params: &TracePairingParams) -> CyclotomicRingElement<F> {
    debug_assert_eq!(a.len(), params.n, "Input vector must have length n = d/k");

    let mut coeffs = vec![F::zero(); params.d];
    let offset = params.d / 2;

    // First half: positions 0..half_n
    coeffs[..params.half_n].copy_from_slice(&a[..params.half_n]);
    // Second half: positions d/2..d/2+half_n
    coeffs[offset..offset + params.half_n].copy_from_slice(&a[params.half_n..2 * params.half_n]);

    CyclotomicRingElement::new(coeffs, params.d)
}

/// Unpack map ψ^{-1}: R_q → F^n
pub fn unpack<F: Field>(x: &CyclotomicRingElement<F>, params: &TracePairingParams) -> Vec<F> {
    debug_assert_eq!(x.dim, params.d);

    let mut a = vec![F::zero(); params.n];
    let offset = params.d / 2;

    a[..params.half_n].copy_from_slice(&x.coeffs[..params.half_n]);
    a[params.half_n..2 * params.half_n].copy_from_slice(&x.coeffs[offset..offset + params.half_n]);

    a
}

// ============================================================================
// Ring Multiplication
// ============================================================================

/// Multiply two ring elements in R_q = Z_q[X]/(X^d + 1)
///
/// Uses schoolbook multiplication with reduction mod (X^d + 1)
pub fn ring_mul<F: Field>(
    a: &CyclotomicRingElement<F>,
    b: &CyclotomicRingElement<F>,
) -> CyclotomicRingElement<F> {
    debug_assert_eq!(a.dim, b.dim);
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
    for (i, &coeff) in result.iter().enumerate() {
        if i < d {
            reduced[i] += coeff;
        } else {
            reduced[i - d] -= coeff;
        }
    }

    CyclotomicRingElement::new(reduced, d)
}

// ============================================================================
// Trace Pairing & Inner Product Recovery
// ============================================================================

/// Trace pairing: Tr_H(a · σ_{-1}(b))
pub fn trace_pairing<F: Field>(
    a: &CyclotomicRingElement<F>,
    b: &CyclotomicRingElement<F>,
    h: &GaloisSubgroup,
) -> CyclotomicRingElement<F> {
    let sigma_neg1_b = b.pow_automorphism(2 * a.dim - 1);
    trace_tower(&ring_mul(a, &sigma_neg1_b), h)
}

/// Recover inner product from packed elements via trace pairing
///
/// **Corollary 3**: Tr_H(ψ(a) · σ_{-1}(ψ(b))) = (d/k) · ⟨a, b⟩
pub fn inner_product_from_trace<F: Field + From<u64>>(
    packed_a: &CyclotomicRingElement<F>,
    packed_b: &CyclotomicRingElement<F>,
    params: &TracePairingParams,
    h: &GaloisSubgroup,
) -> F {
    let trace_result = trace_pairing(packed_a, packed_b, h);
    let scale_inv = F::from(params.scale as u64)
        .inverse()
        .expect("scale must be invertible");
    trace_result.coeffs[0] * scale_inv
}

/// Compute inner product directly (for verification)
pub fn direct_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(ai, bi)| *ai * *bi).sum()
}

/// Verify basis orthogonality: Tr_H(e_a · σ_{-1}(e_b)) = (d/k) · δ_{ab}
pub fn verify_basis_orthogonality<F: Field + From<u64>>(
    params: &TracePairingParams,
    h: &GaloisSubgroup,
) -> bool {
    let scale = F::from(params.scale as u64);
    let check_limit = params.n.min(8);

    (0..check_limit).all(|a_idx| {
        (0..check_limit).all(|b_idx| {
            let e_a = pack(&basis_vector::<F>(a_idx, params.n), params);
            let e_b = pack(&basis_vector::<F>(b_idx, params.n), params);
            let result = trace_pairing(&e_a, &e_b, h);
            let expected = if a_idx == b_idx { scale } else { F::zero() };
            result.coeffs[0] == expected
        })
    })
}

/// Create basis vector e_i (1 at position i, 0 elsewhere)
fn basis_vector<F: Field>(i: usize, n: usize) -> Vec<F> {
    let mut v = vec![F::zero(); n];
    v[i] = F::one();
    v
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_local_definitions)]
mod test_config {
    use ark_ff::fields::models::fp::{Fp64, MontBackend, MontConfig};

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
        let params = TracePairingParams::new(64, 4);
        let a: Vec<TestFq> = (0..params.n as u64).map(TestFq::from).collect();

        assert_eq!(a, unpack(&pack(&a, &params), &params));
    }

    #[test]
    fn test_ring_mul_identity() {
        let d = 8;
        let one = CyclotomicRingElement::<TestFq>::new(vec![TestFq::from(1u64)], d);
        let x = CyclotomicRingElement::<TestFq>::new(
            vec![TestFq::from(1u64), TestFq::from(2u64), TestFq::from(3u64)],
            d,
        );
        assert_eq!(ring_mul(&one, &x).coeffs, x.coeffs);
    }

    #[test]
    fn test_ring_mul_x_squared() {
        let d = 4;
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
        assert_eq!(x_squared.coeffs[2], TestFq::from(1u64));
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
        assert_eq!(result.coeffs[0], -TestFq::from(1u64));
    }

    #[test]
    fn test_inner_product_recovery_small() {
        let params = TracePairingParams::new(16, 2);
        let h = GaloisSubgroup::new(params.d, params.k);

        let a: Vec<TestFq> = [1, 2, 3, 0, 0, 0, 0, 0]
            .map(|x| TestFq::from(x as u64))
            .to_vec();
        let b: Vec<TestFq> = [4, 5, 6, 0, 0, 0, 0, 0]
            .map(|x| TestFq::from(x as u64))
            .to_vec();

        let direct = direct_inner_product(&a, &b);
        assert_eq!(direct, TestFq::from(32u64)); // 4 + 10 + 18

        let recovered =
            inner_product_from_trace(&pack(&a, &params), &pack(&b, &params), &params, &h);
        assert_eq!(recovered, direct);
    }

    #[test]
    fn test_inner_product_recovery_random() {
        let params = TracePairingParams::new(32, 4);
        let h = GaloisSubgroup::new(params.d, params.k);
        let mut rng = rand::thread_rng();

        for _ in 0..5 {
            let a: Vec<TestFq> = (0..params.n).map(|_| TestFq::rand(&mut rng)).collect();
            let b: Vec<TestFq> = (0..params.n).map(|_| TestFq::rand(&mut rng)).collect();

            let direct = direct_inner_product(&a, &b);
            let recovered =
                inner_product_from_trace(&pack(&a, &params), &pack(&b, &params), &params, &h);
            assert_eq!(recovered, direct);
        }
    }

    #[test]
    fn test_basis_orthogonality() {
        let params = TracePairingParams::new(16, 2);
        let h = GaloisSubgroup::new(params.d, params.k);
        assert!(verify_basis_orthogonality::<TestFq>(&params, &h));
    }

    #[test]
    fn test_hachi_params() {
        let params = TracePairingParams::hachi();
        assert_eq!(
            (params.d, params.k, params.n, params.half_n, params.scale),
            (1024, 4, 256, 128, 256)
        );
    }

    #[test]
    fn test_trace_pairing_linearity() {
        let params = TracePairingParams::new(16, 2);
        let h = GaloisSubgroup::new(params.d, params.k);

        let a1: Vec<TestFq> = [1, 0, 0, 0, 0, 0, 0, 0]
            .map(|x| TestFq::from(x as u64))
            .to_vec();
        let a2: Vec<TestFq> = [0, 1, 0, 0, 0, 0, 0, 0]
            .map(|x| TestFq::from(x as u64))
            .to_vec();
        let b: Vec<TestFq> = [3, 4, 0, 0, 0, 0, 0, 0]
            .map(|x| TestFq::from(x as u64))
            .to_vec();

        let (pa1, pa2, pb) = (pack(&a1, &params), pack(&a2, &params), pack(&b, &params));

        let ip1 = inner_product_from_trace(&pa1, &pb, &params, &h);
        let ip2 = inner_product_from_trace(&pa2, &pb, &params, &h);

        // a1 + a2
        let a_sum: Vec<TestFq> = a1.iter().zip(&a2).map(|(x, y)| *x + *y).collect();
        let ip_sum = inner_product_from_trace(&pack(&a_sum, &params), &pb, &params, &h);

        assert_eq!(ip_sum, ip1 + ip2);
    }
}
