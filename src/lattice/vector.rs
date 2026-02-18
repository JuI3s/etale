//! Vector operations over Z_q

use rand::Rng;
use std::fmt;

// ============================================================================
// Vector over Z_q
// ============================================================================

/// A vector in Z_q^n
#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    pub coeffs: Vec<i64>,
    pub modulus: i64,
}

impl Vector {
    /// Create new vector, reducing coefficients mod q
    pub fn new(coeffs: Vec<i64>, modulus: i64) -> Self {
        let reduced = coeffs
            .into_iter()
            .map(|c| c.rem_euclid(modulus))
            .collect();
        Self { coeffs: reduced, modulus }
    }

    /// Create zero vector of length n
    pub fn zero(n: usize, modulus: i64) -> Self {
        Self { coeffs: vec![0; n], modulus }
    }

    /// Create random vector with coefficients in [-bound, bound]
    pub fn random<R: Rng>(rng: &mut R, n: usize, bound: i64, modulus: i64) -> Self {
        let coeffs = (0..n).map(|_| rng.gen_range(-bound..=bound)).collect();
        Self::new(coeffs, modulus)
    }

    /// Create ternary vector (coefficients in {-1, 0, 1})
    pub fn random_ternary<R: Rng>(rng: &mut R, n: usize, modulus: i64) -> Self {
        Self::random(rng, n, 1, modulus)
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }
}

// ============================================================================
// Centered Representation & Norms
// ============================================================================

impl Vector {
    /// Center coefficient to [-q/2, q/2]
    #[inline]
    fn center(&self, c: i64) -> i64 {
        if c > self.modulus / 2 { c - self.modulus } else { c }
    }

    /// Centered representative (coefficients in [-q/2, q/2])
    pub fn centered(&self) -> Vec<i64> {
        self.coeffs.iter().map(|&c| self.center(c)).collect()
    }

    /// ℓ∞ norm (max absolute value of centered representatives)
    pub fn ell_inf_norm(&self) -> i64 {
        self.coeffs
            .iter()
            .map(|&c| self.center(c).abs())
            .max()
            .unwrap_or(0)
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

impl Vector {
    /// Add two vectors
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.coeffs.len(), other.coeffs.len());
        debug_assert_eq!(self.modulus, other.modulus);

        let coeffs = self.coeffs.iter()
            .zip(&other.coeffs)
            .map(|(&a, &b)| (a + b).rem_euclid(self.modulus))
            .collect();

        Self { coeffs, modulus: self.modulus }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: i64) -> Self {
        Self::new(
            self.coeffs.iter().map(|&c| c * scalar).collect(),
            self.modulus,
        )
    }

    /// Inner product: ⟨self, other⟩ mod q
    pub fn inner_product(&self, other: &Self) -> i64 {
        debug_assert_eq!(self.coeffs.len(), other.coeffs.len());

        self.coeffs.iter()
            .zip(&other.coeffs)
            .map(|(&a, &b)| a * b)
            .sum::<i64>()
            .rem_euclid(self.modulus)
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (mod {})", self.centered(), self.modulus)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_operations() {
        let q = 7681;
        let v1 = Vector::new(vec![1, 2, 3, 4], q);
        let v2 = Vector::new(vec![5, 6, 7, 8], q);

        assert_eq!(v1.add(&v2).coeffs, vec![6, 8, 10, 12]);
        assert_eq!(v1.scalar_mul(2).coeffs, vec![2, 4, 6, 8]);
        assert_eq!(v1.inner_product(&v2), 70); // 5 + 12 + 21 + 32
    }

    #[test]
    fn test_vector_linf_norm() {
        let q = 101;

        // 100 mod 101 = -1 centered, 50 stays 50
        let v = Vector::new(vec![100, 50, 1], q);
        assert_eq!(v.ell_inf_norm(), 50);

        let v2 = Vector::new(vec![-10, 20, -30], q);
        assert_eq!(v2.ell_inf_norm(), 30);
    }

    #[test]
    fn test_rem_euclid_signed_values() {
        let q = 17;

        // Negative values are properly reduced via rem_euclid
        // -1.rem_euclid(17) = 16, -5.rem_euclid(17) = 12
        let v = Vector::new(vec![-1, -5, -17, -18], q);
        assert_eq!(v.coeffs, vec![16, 12, 0, 16]);

        // Addition with signed overflow
        let v1 = Vector::new(vec![15, 10], q);
        let v2 = Vector::new(vec![5, 10], q);
        let sum = v1.add(&v2);
        // (15+5) % 17 = 3, (10+10) % 17 = 3
        assert_eq!(sum.coeffs, vec![3, 3]);

        // Inner product with large intermediate values
        let a = Vector::new(vec![10, 10], q);
        let b = Vector::new(vec![10, 10], q);
        // 10*10 + 10*10 = 200, 200.rem_euclid(17) = 200 % 17 = 13
        assert_eq!(a.inner_product(&b), 13);

        // Verify rem_euclid vs % operator difference
        assert_eq!((-5i64).rem_euclid(17), 12);
        assert_eq!((-5i64) % 17, -5);
    }
}
