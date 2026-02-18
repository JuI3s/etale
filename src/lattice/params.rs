//! Parameter sets for lattice-based polynomial commitment schemes
//!
//! This module defines parameter sets for various lattice-based cryptographic schemes,
//! including Hachi, Greyhound, and NIST standards (Kyber, Dilithium, Falcon).
//!
//! # Parameter Sets
//!
//! | Name | d | q | Use Case |
//! |------|---|---|----------|
//! | `HACHI` | 1024 | 2³² | Hachi polynomial commitment |
//! | `COMPRESSED_K*` | varies | 2³² | Trace compression research |
//! | `GREYHOUND` | 64 | 2³² | Greyhound baseline |
//! | `DILITHIUM_2` | 256 | ~2²³ | NIST ML-DSA (NTT-friendly) |
//! | `FALCON_512` | 512 | 12289 | NIST Falcon (NTT-friendly) |
//!
//! # References
//!
//! - [Hachi](https://eprint.iacr.org/2026/156) - DQZZ26
//! - Greyhound (CRYPTO 2024) - NS24
//! - NIST FIPS 203/204/206

use super::modular::pow_mod;

// ============================================================================
// Parameter Set Structure
// ============================================================================

/// Parameter set for lattice-based polynomial commitment schemes
#[derive(Clone, Debug)]
pub struct RingParams {
    /// Ring dimension d = 2^α (polynomial degree < d)
    pub d: usize,
    /// Modulus q (prime)
    pub q: u64,
    /// Module rank κ (n_A = n_B = n_D in Hachi)
    pub kappa: usize,
    /// Decomposition base b
    pub b: u64,
    /// Expansion factor δ = ⌈log_b(q)⌉
    pub delta: usize,
    /// Infinity norm bound for decomposed elements (b - 1)
    pub beta_inf: u64,
    /// Human-readable name
    pub name: &'static str,
}

impl RingParams {
    /// Check if this parameter set supports direct negacyclic NTT.
    ///
    /// For R_q = Z_q[X]/(X^d + 1), direct NTT requires q ≡ 1 (mod 2d),
    /// which ensures primitive 2d-th roots of unity exist in Z_q.
    pub fn supports_direct_ntt(&self) -> bool {
        self.q % (2 * self.d as u64) == 1
    }

    /// Find a primitive 2d-th root of unity mod q, if one exists.
    ///
    /// Returns Some(ψ) where ψ^{2d} = 1 and ψ^d = -1 (mod q), or None.
    pub fn find_primitive_root(&self) -> Option<u64> {
        if !self.supports_direct_ntt() {
            return None;
        }
        find_primitive_2d_root(self.q, self.d)
    }

    /// Verify that a given ψ is a valid primitive 2d-th root of unity.
    pub fn verify_primitive_root(&self, psi: u64) -> bool {
        let two_d = 2 * self.d as u64;
        pow_mod(psi, two_d, self.q) == 1 && pow_mod(psi, self.d as u64, self.q) == self.q - 1
    }

    /// Compute δ = ⌈log_b(q)⌉
    pub fn compute_delta(q: u64, b: u64) -> usize {
        let mut delta = 0;
        let mut power = 1u64;
        while power < q {
            power = power.saturating_mul(b);
            delta += 1;
        }
        delta
    }

    /// Create a compressed parameter set for trace compression research.
    ///
    /// Given a compression factor k, creates params with d' = base_d / k.
    /// The beta_inf is scaled by k (worst-case) for trace amplification.
    pub fn compressed(base: &RingParams, k: usize, name: &'static str) -> Self {
        assert!(k.is_power_of_two());
        assert!(base.d >= k);
        Self {
            d: base.d / k,
            q: base.q,
            kappa: base.kappa,
            b: base.b,
            delta: base.delta,
            beta_inf: base.beta_inf * k as u64, // worst-case trace amplification
            name,
        }
    }
}

/// Find a primitive 2d-th root of unity mod q.
///
/// Requires q ≡ 1 (mod 2d). Returns ψ such that ψ^{2d} = 1 and ψ^d = -1 (mod q).
pub fn find_primitive_2d_root(q: u64, d: usize) -> Option<u64> {
    let two_d = 2 * d as u64;
    if q % two_d != 1 {
        return None;
    }

    // Find a generator of Z_q^* and compute ψ = g^{(q-1)/(2d)}
    // This gives a primitive 2d-th root since ord(ψ) = 2d
    let exp = (q - 1) / two_d;

    // Try small candidates for generator
    for g in 2..q {
        let psi = pow_mod(g, exp, q);
        // Verify: ψ^{2d} = 1 and ψ^d = -1 (i.e., ψ^d = q-1)
        if pow_mod(psi, two_d, q) == 1 && pow_mod(psi, d as u64, q) == q - 1 {
            return Some(psi);
        }
    }
    None
}

// ============================================================================
// Hachi Parameters [DQZZ26, Figure 9]
// ============================================================================

/// Hachi polynomial commitment scheme parameters
///
/// - d = 1024, q ≈ 2³² (non-NTT-friendly)
/// - Uses base-16 decomposition with δ = 8
///
/// Reference: [Hachi](https://eprint.iacr.org/2026/156), Figure 9
pub const HACHI: RingParams = RingParams {
    d: 1024,
    q: 4294967197, // ≈ 2^32, prime, q ≢ 1 (mod 2048)
    kappa: 1,
    b: 16,
    delta: 8, // ⌈log_16(2^32)⌉ = 8
    beta_inf: 15,
    name: "Hachi",
};

// ============================================================================
// Compressed Parameter Sets (Trace Compression Research)
// ============================================================================

/// Compression factor k=4: d=256
///
/// Worst-case trace amplification: beta_inf = 4 * 15 = 60
pub const COMPRESSED_K4: RingParams = RingParams {
    d: 256,
    q: 4294967197,
    kappa: 1,
    b: 16,
    delta: 8,
    beta_inf: 60,
    name: "Compressed_k4",
};

/// Compression factor k=8: d=128
///
/// Worst-case trace amplification: beta_inf = 8 * 15 = 120
pub const COMPRESSED_K8: RingParams = RingParams {
    d: 128,
    q: 4294967197,
    kappa: 1,
    b: 16,
    delta: 8,
    beta_inf: 120,
    name: "Compressed_k8",
};

/// Compression factor k=16: d=64 (THE SWEET SPOT)
///
/// Worst-case trace amplification: beta_inf = 16 * 15 = 240
/// This is still less than Greyhound's 255!
pub const COMPRESSED_K16: RingParams = RingParams {
    d: 64,
    q: 4294967197,
    kappa: 1,
    b: 16,
    delta: 8,
    beta_inf: 240,
    name: "Compressed_k16",
};

/// Compression factor k=32: d=32 (aggressive)
///
/// Worst-case trace amplification: beta_inf = 32 * 15 = 480
pub const COMPRESSED_K32: RingParams = RingParams {
    d: 32,
    q: 4294967197,
    kappa: 1,
    b: 16,
    delta: 8,
    beta_inf: 480,
    name: "Compressed_k32",
};

// ============================================================================
// Greyhound Baseline [NS24, Section 5]
// ============================================================================

/// Greyhound polynomial commitment scheme parameters
///
/// Reference: Greyhound (CRYPTO 2024), Section 5, Table 1
pub const GREYHOUND: RingParams = RingParams {
    d: 64,
    q: 4294967197,
    kappa: 1,
    b: 256,
    delta: 4, // ⌈log_256(2^32)⌉ = 4
    beta_inf: 255,
    name: "Greyhound",
};

// ============================================================================
// NIST Standards
// ============================================================================

/// NIST ML-KEM-512 (Kyber) parameters [FIPS 203]
///
/// Note: q = 3329 ≡ 1 (mod 256), but NOT ≡ 1 (mod 512).
/// Kyber uses a modified NTT that factors X^256+1 into degree-2 polynomials.
/// `supports_direct_ntt()` correctly returns false for standard negacyclic NTT.
pub const KYBER_512: RingParams = RingParams {
    d: 256,
    q: 3329,
    kappa: 2,
    b: 2,
    delta: 12,
    beta_inf: 1,
    name: "Kyber_512",
};

/// NIST ML-DSA-44 (Dilithium) parameters [FIPS 204]
///
/// q = 8380417 ≡ 1 (mod 512), supports direct negacyclic NTT.
/// Primitive root: ψ = 1753
pub const DILITHIUM_2: RingParams = RingParams {
    d: 256,
    q: 8380417,
    kappa: 4,
    b: 2,
    delta: 23,
    beta_inf: 1,
    name: "Dilithium_2",
};

/// NIST Falcon-512 parameters [FIPS 206]
///
/// q = 12289 ≡ 1 (mod 1024), supports direct negacyclic NTT.
/// Primitive root: ψ = 49
pub const FALCON_512: RingParams = RingParams {
    d: 512,
    q: 12289,
    kappa: 1,
    b: 2,
    delta: 14,
    beta_inf: 1,
    name: "Falcon_512",
};

// ============================================================================
// Parameter Collections
// ============================================================================

/// All parameter sets for benchmarking
pub const ALL_PARAMS: &[&RingParams] = &[
    &HACHI,
    &COMPRESSED_K4,
    &COMPRESSED_K8,
    &COMPRESSED_K16,
    &COMPRESSED_K32,
    &GREYHOUND,
    &KYBER_512,
    &DILITHIUM_2,
    &FALCON_512,
];

/// Hachi-based parameter sets (same q, varying d)
pub const HACHI_FAMILY: &[&RingParams] = &[
    &HACHI,
    &COMPRESSED_K4,
    &COMPRESSED_K8,
    &COMPRESSED_K16,
    &COMPRESSED_K32,
];

/// NTT-friendly parameter sets (q ≡ 1 mod 2d)
pub const NTT_FRIENDLY: &[&RingParams] = &[
    &DILITHIUM_2,
    &FALCON_512,
];

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_direct_ntt() {
        // Hachi: q = 4294967197, d = 1024 → 2d = 2048
        // 4294967197 % 2048 = 1949 ≠ 1
        assert!(!HACHI.supports_direct_ntt());

        // Kyber: q = 3329, d = 256 → 2d = 512
        // 3329 % 512 = 257 ≠ 1
        assert!(!KYBER_512.supports_direct_ntt());

        // Dilithium: q = 8380417, d = 256 → 2d = 512
        // 8380417 % 512 = 1 ✓
        assert!(DILITHIUM_2.supports_direct_ntt());

        // Falcon: q = 12289, d = 512 → 2d = 1024
        // 12289 % 1024 = 1 ✓
        assert!(FALCON_512.supports_direct_ntt());

        // Compressed params all use same q as Hachi
        assert!(!COMPRESSED_K4.supports_direct_ntt());
        assert!(!COMPRESSED_K16.supports_direct_ntt());
        assert!(!GREYHOUND.supports_direct_ntt());
    }

    #[test]
    fn test_find_primitive_root() {
        // Non-NTT-friendly should return None
        assert!(HACHI.find_primitive_root().is_none());
        assert!(KYBER_512.find_primitive_root().is_none());

        // Dilithium should find a valid root
        let psi_dil = DILITHIUM_2.find_primitive_root();
        assert!(psi_dil.is_some());
        assert!(DILITHIUM_2.verify_primitive_root(psi_dil.unwrap()));
        assert!(DILITHIUM_2.verify_primitive_root(1753));

        // Falcon should find a valid root
        let psi_fal = FALCON_512.find_primitive_root();
        assert!(psi_fal.is_some());
        assert!(FALCON_512.verify_primitive_root(psi_fal.unwrap()));
        assert!(FALCON_512.verify_primitive_root(49));
    }

    #[test]
    fn test_verify_primitive_root() {
        assert!(DILITHIUM_2.verify_primitive_root(1753));
        assert!(FALCON_512.verify_primitive_root(49));
        assert!(!DILITHIUM_2.verify_primitive_root(1));
        assert!(!DILITHIUM_2.verify_primitive_root(2));
    }

    #[test]
    fn test_compute_delta() {
        assert_eq!(RingParams::compute_delta(4294967197, 16), 8); // Hachi
        assert_eq!(RingParams::compute_delta(4294967197, 256), 4); // Greyhound
        assert_eq!(RingParams::compute_delta(3329, 2), 12); // Kyber
    }

    #[test]
    fn test_compressed_constructor() {
        let c4 = RingParams::compressed(&HACHI, 4, "test_k4");
        assert_eq!(c4.d, 256);
        assert_eq!(c4.q, HACHI.q);
        assert_eq!(c4.beta_inf, 60); // 4 * 15

        let c16 = RingParams::compressed(&HACHI, 16, "test_k16");
        assert_eq!(c16.d, 64);
        assert_eq!(c16.beta_inf, 240); // 16 * 15
    }

    #[test]
    fn test_param_collections() {
        assert_eq!(ALL_PARAMS.len(), 9);
        assert_eq!(HACHI_FAMILY.len(), 5);
        assert_eq!(NTT_FRIENDLY.len(), 2);

        // All NTT_FRIENDLY should support direct NTT
        for p in NTT_FRIENDLY {
            assert!(p.supports_direct_ntt(), "{} should be NTT-friendly", p.name);
        }
    }
}

