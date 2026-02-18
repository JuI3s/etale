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

/// Decompose integer a into base-b digits
///
/// Returns δ digits in {0, ..., b-1} such that a = Σ digits[i] · b^i
#[inline]
fn decompose_scalar(a: u64, b: u64, delta: usize) -> Vec<u64> {
    let mut digits = Vec::with_capacity(delta);
    let mut rem = a;
    for _ in 0..delta {
        digits.push(rem % b);
        rem /= b;
    }
    digits
}

/// Decompose polynomial coefficient-wise
///
/// Returns δ polynomials, each with coefficients in {0, ..., b-1}
///
/// # Example
///
/// For f with coefficients in [0, q) and b=16, δ=8:
/// - f = f_0 + f_1·16 + f_2·256 + ... + f_7·16^7
/// - Each f_i has coefficients in {0, ..., 15}
pub fn decompose_poly(f: &RingElement, b: u64, delta: usize) -> Vec<RingElement> {
    let d = f.d;
    let q = f.q;
    let mut result: Vec<Vec<u64>> = vec![vec![0; d]; delta];

    for i in 0..d {
        let digits = decompose_scalar(f.coeffs[i], b, delta);
        for j in 0..delta {
            result[j][i] = digits[j];
        }
    }

    result
        .into_iter()
        .map(|coeffs| RingElement { coeffs, d, q })
        .collect()
}

/// Recompose polynomial from base-b digits
///
/// Inverse of decompose_poly: f = Σ decomposed[i] · b^i
pub fn recompose_poly(decomposed: &[RingElement], b: u64) -> RingElement {
    assert!(!decomposed.is_empty());
    let d = decomposed[0].d;
    let q = decomposed[0].q;

    let mut result = vec![0u64; d];
    let mut power = 1u64;

    for component in decomposed {
        assert_eq!(component.d, d);
        for i in 0..d {
            result[i] = add_mod(result[i], mul_mod(component.coeffs[i], power, q), q);
        }
        power = mul_mod(power, b, q);
    }

    RingElement { coeffs: result, d, q }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(12345)
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

        // Check that decomposed elements have small coefficients
        for component in &decomposed {
            for &c in &component.coeffs {
                assert!(c < b, "Decomposed coefficient {} >= base {}", c, b);
            }
        }

        // Check roundtrip
        let recomposed = recompose_poly(&decomposed, b);
        assert_eq!(f.coeffs, recomposed.coeffs);
    }

    #[test]
    fn test_decompose_scalar() {
        // 255 = 15 + 15*16 in base 16
        let digits = decompose_scalar(255, 16, 2);
        assert_eq!(digits, vec![15, 15]);

        // 256 = 0 + 0*16 + 1*256 in base 16
        let digits = decompose_scalar(256, 16, 3);
        assert_eq!(digits, vec![0, 0, 1]);
    }
}

