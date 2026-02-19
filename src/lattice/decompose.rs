//! Base-b decomposition for gadget-based lattice commitments
//!
//! Gadget decomposition splits ring elements into low-norm components:
//! f = Σ_{i=0}^{δ-1} f_i · b^i where each f_i has coefficients in {0, ..., b-1}
//!
//! This is used in:
//! - Hachi polynomial commitments (b=16, δ=8)
//! - Greyhound commitments (b=256, δ=4)
//! - General gadget-based constructions
//!
//! # References
//!
//! - [Hachi](https://eprint.iacr.org/2026/156) - DQZZ26

use super::modular::{add_mod, mul_mod};
use super::ntt::RingElement;

// ============================================================================
// Scalar Decomposition
// ============================================================================

/// Decompose integer into base-b digits.
///
/// Returns δ digits in {0, ..., b-1} such that a = Σ digits[i] · b^i
#[inline]
fn decompose_scalar(mut a: u64, b: u64, delta: usize) -> Vec<u64> {
    (0..delta)
        .map(|_| {
            let digit = a % b;
            a /= b;
            digit
        })
        .collect()
}

// ============================================================================
// Polynomial Decomposition
// ============================================================================

/// Decompose polynomial coefficient-wise into base-b components.
///
/// Returns δ polynomials, each with coefficients in {0, ..., b-1}.
///
/// # Example
///
/// For f with coefficients in [0, q) and b=16, δ=8:
/// - f = f_0 + f_1·16 + f_2·256 + ... + f_7·16^7
/// - Each f_i has coefficients in {0, ..., 15}
pub fn decompose_poly(f: &RingElement, b: u64, delta: usize) -> Vec<RingElement> {
    let (d, q) = (f.d, f.q);

    // Decompose each coefficient, then transpose to get δ polynomials
    let all_digits: Vec<_> = f
        .coeffs
        .iter()
        .map(|&c| decompose_scalar(c, b, delta))
        .collect();

    (0..delta)
        .map(|j| {
            let coeffs = all_digits.iter().map(|digits| digits[j]).collect();
            RingElement { coeffs, d, q }
        })
        .collect()
}

/// Recompose polynomial from base-b digit polynomials.
///
/// Inverse of decompose_poly: f = Σ decomposed[i] · b^i
pub fn recompose_poly(decomposed: &[RingElement], b: u64) -> RingElement {
    let first = decomposed.first().expect("decomposed cannot be empty");
    let (d, q) = (first.d, first.q);

    // Fold over components, accumulating with powers of b
    let coeffs = decomposed
        .iter()
        .enumerate()
        .fold(vec![0u64; d], |mut acc, (i, component)| {
            debug_assert_eq!(component.d, d, "dimension mismatch");
            let power = pow_of_b(b, i, q);
            acc.iter_mut()
                .zip(&component.coeffs)
                .for_each(|(a, &c)| *a = add_mod(*a, mul_mod(c, power, q), q));
            acc
        });

    RingElement { coeffs, d, q }
}

/// Compute b^exp mod q efficiently.
#[inline]
fn pow_of_b(b: u64, exp: usize, q: u64) -> u64 {
    (0..exp).fold(1u64, |acc, _| mul_mod(acc, b, q))
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
    fn test_decompose_scalar() {
        // 255 = 15 + 15*16 in base 16
        assert_eq!(decompose_scalar(255, 16, 2), vec![15, 15]);

        // 256 = 0 + 0*16 + 1*256 in base 16
        assert_eq!(decompose_scalar(256, 16, 3), vec![0, 0, 1]);
    }

    #[test]
    fn test_decompose_recompose() {
        let mut rng = test_rng();
        let d = 64;
        let q = (1u64 << 32) - 5; // Hachi modulus
        let b = 16u64;
        let delta = 8;

        let f = RingElement::random(&mut rng, d, q);
        let decomposed = decompose_poly(&f, b, delta);

        // Verify small coefficients
        for component in &decomposed {
            assert!(
                component.coeffs.iter().all(|&c| c < b),
                "Decomposed coefficient exceeds base"
            );
        }

        // Verify roundtrip
        let recomposed = recompose_poly(&decomposed, b);
        assert_eq!(f.coeffs, recomposed.coeffs);
    }

    #[test]
    fn test_decompose_single_coeff() {
        let q = 1000;
        let b = 10u64;
        let delta = 3;

        // 123 = 3 + 2*10 + 1*100
        let f = RingElement::new(vec![123], 1, q);
        let decomposed = decompose_poly(&f, b, delta);

        assert_eq!(decomposed[0].coeffs[0], 3);
        assert_eq!(decomposed[1].coeffs[0], 2);
        assert_eq!(decomposed[2].coeffs[0], 1);
    }
}
