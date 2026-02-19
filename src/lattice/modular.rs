//! Modular arithmetic operations for Z_q
//!
//! This module provides efficient modular arithmetic primitives used throughout
//! lattice-based cryptographic operations.

/// Add two values modulo q
///
/// Assumes a, b < q. Returns (a + b) mod q.
#[inline(always)]
pub fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    let sum = a + b;
    if sum >= q {
        sum - q
    } else {
        sum
    }
}

/// Subtract two values modulo q
///
/// Assumes a, b < q. Returns (a - b) mod q.
#[inline(always)]
pub fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        q - b + a
    }
}

/// Multiply two values modulo q
///
/// Uses 128-bit intermediate to avoid overflow.
#[inline(always)]
pub fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// Negate a value modulo q
///
/// Returns -a mod q = q - a (or 0 if a == 0).
#[inline(always)]
pub fn neg_mod(a: u64, q: u64) -> u64 {
    if a == 0 {
        0
    } else {
        q - a
    }
}

/// Compute modular inverse using extended Euclidean algorithm
///
/// Returns a^{-1} mod q such that a * a^{-1} ≡ 1 (mod q).
/// Panics if gcd(a, q) ≠ 1.
pub fn mod_inv(a: u64, q: u64) -> u64 {
    let (mut old_r, mut r) = (a as i128, q as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        let temp = r;
        r = old_r - quotient * r;
        old_r = temp;
        let temp = s;
        s = old_s - quotient * s;
        old_s = temp;
    }

    debug_assert_eq!(old_r, 1, "gcd(a, q) must be 1 for inverse to exist");
    ((old_s % q as i128 + q as i128) % q as i128) as u64
}

/// Compute base^exp mod modulus using binary exponentiation
///
/// Time complexity: O(log exp)
pub fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mul_mod(result, base, modulus);
        }
        exp /= 2;
        base = mul_mod(base, base, modulus);
    }
    result
}

/// Reduce a signed value to [0, q)
#[inline(always)]
pub fn reduce_signed(a: i64, q: u64) -> u64 {
    ((a % q as i64) + q as i64) as u64 % q
}

/// Reduce a 128-bit value to [0, q)
#[inline(always)]
pub fn reduce_i128(a: i128, q: u64) -> u64 {
    ((a % q as i128 + q as i128) % q as i128) as u64
}

// ============================================================================
// Arithmetic context (optional convenience wrapper)
// ============================================================================

/// Modular arithmetic context for Z_q
///
/// Provides cleaner syntax when doing many operations with the same modulus.
///
/// # Example
/// ```
/// use etale::lattice::modular::Zq;
///
/// let zq = Zq::new(17);
/// let a = 10u64;
/// let b = 12u64;
/// assert_eq!(zq.add(a, b), 5);  // (10 + 12) mod 17 = 5
/// assert_eq!(zq.mul(a, b), 1);  // (10 * 12) mod 17 = 1
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zq {
    pub q: u64,
}

impl Zq {
    /// Create a new modular arithmetic context for Z_q
    #[inline]
    pub const fn new(q: u64) -> Self {
        Self { q }
    }

    /// Add two values: (a + b) mod q
    #[inline(always)]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        add_mod(a, b, self.q)
    }

    /// Subtract two values: (a - b) mod q
    #[inline(always)]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        sub_mod(a, b, self.q)
    }

    /// Multiply two values: (a * b) mod q
    #[inline(always)]
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        mul_mod(a, b, self.q)
    }

    /// Negate a value: -a mod q
    #[inline(always)]
    pub fn neg(&self, a: u64) -> u64 {
        neg_mod(a, self.q)
    }

    /// Compute modular inverse: a^{-1} mod q
    #[inline]
    pub fn inv(&self, a: u64) -> u64 {
        mod_inv(a, self.q)
    }

    /// Compute power: base^exp mod q
    #[inline]
    pub fn pow(&self, base: u64, exp: u64) -> u64 {
        pow_mod(base, exp, self.q)
    }

    /// Reduce a signed value to [0, q)
    #[inline(always)]
    pub fn from_signed(&self, a: i64) -> u64 {
        reduce_signed(a, self.q)
    }

    /// Reduce a 128-bit value to [0, q)
    #[inline(always)]
    pub fn from_i128(&self, a: i128) -> u64 {
        reduce_i128(a, self.q)
    }

    /// Center a value to [-q/2, q/2)
    #[inline]
    pub fn center(&self, a: u64) -> i64 {
        let half_q = self.q as i64 / 2;
        let a = a as i64;
        if a > half_q {
            a - self.q as i64
        } else {
            a
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_mod() {
        let q = 17u64;
        assert_eq!(add_mod(10, 10, q), 3); // 20 mod 17 = 3
        assert_eq!(add_mod(0, 0, q), 0);
        assert_eq!(add_mod(16, 1, q), 0); // 17 mod 17 = 0
        assert_eq!(add_mod(8, 8, q), 16);
    }

    #[test]
    fn test_sub_mod() {
        let q = 17u64;
        assert_eq!(sub_mod(10, 5, q), 5);
        assert_eq!(sub_mod(5, 10, q), 12); // -5 mod 17 = 12
        assert_eq!(sub_mod(0, 1, q), 16);
        assert_eq!(sub_mod(0, 0, q), 0);
    }

    #[test]
    fn test_mul_mod() {
        let q = 17u64;
        assert_eq!(mul_mod(5, 7, q), 1); // 35 mod 17 = 1
        assert_eq!(mul_mod(0, 10, q), 0);
        assert_eq!(mul_mod(3, 6, q), 1); // 18 mod 17 = 1

        // Test no overflow for large values
        let big_q = 4_294_967_197_u64;
        let a = 4_000_000_000_u64;
        let b = 3_000_000_000_u64;
        let result = mul_mod(a, b, big_q);
        assert!(result < big_q);
    }

    #[test]
    fn test_neg_mod() {
        let q = 17u64;
        assert_eq!(neg_mod(5, q), 12); // -5 mod 17 = 12
        assert_eq!(neg_mod(0, q), 0);
        assert_eq!(neg_mod(1, q), 16);
    }

    #[test]
    fn test_mod_inv() {
        let q = 17u64;
        // 5 * 7 = 35 ≡ 1 (mod 17)
        assert_eq!(mod_inv(5, q), 7);
        assert_eq!(mul_mod(5, mod_inv(5, q), q), 1);

        // Test for various values
        for a in 1..q {
            let inv = mod_inv(a, q);
            assert_eq!(mul_mod(a, inv, q), 1, "Failed for a={a}");
        }
    }

    #[test]
    fn test_pow_mod() {
        let q = 17u64;
        assert_eq!(pow_mod(3, 0, q), 1);
        assert_eq!(pow_mod(3, 1, q), 3);
        assert_eq!(pow_mod(3, 2, q), 9);
        assert_eq!(pow_mod(3, 4, q), 13); // 81 mod 17 = 13
        assert_eq!(pow_mod(3, 16, q), 1); // Fermat's little theorem

        // Large exponent
        let big_q = 8_380_417_u64;
        assert_eq!(pow_mod(1753, 512, big_q), 1); // primitive 512th root
    }

    #[test]
    fn test_zq_context() {
        let zq = Zq::new(17);

        assert_eq!(zq.add(10, 10), 3);
        assert_eq!(zq.sub(5, 10), 12);
        assert_eq!(zq.mul(5, 7), 1);
        assert_eq!(zq.neg(5), 12);
        assert_eq!(zq.inv(5), 7);
        assert_eq!(zq.pow(3, 4), 13);
        assert_eq!(zq.from_signed(-5), 12);
        assert_eq!(zq.center(12), -5);
    }

    #[test]
    fn test_reduce_signed() {
        let q = 17u64;
        assert_eq!(reduce_signed(5, q), 5);
        assert_eq!(reduce_signed(-5, q), 12);
        assert_eq!(reduce_signed(-17, q), 0);
        assert_eq!(reduce_signed(17, q), 0);
        assert_eq!(reduce_signed(34, q), 0);
    }
}
