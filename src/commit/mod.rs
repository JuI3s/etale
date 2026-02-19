//! Lattice-based polynomial commitment schemes
//!
//! This module provides Ajtai/SIS-based commitment schemes for polynomial vectors.
//!
//! # Core Interface
//!
//! [`AjtaiKey<R>`] is a random matrix A ∈ R_q^{κ × m}, generic over ring
//! representation. The commitment to a short vector s is `t = A · s`.
//!
//! - `AjtaiKey<RingElement>`: coefficient domain, O(d²) mul
//! - `AjtaiKey<NttRingElement>`: NTT domain, O(d) mul
//!
//! Use [`AjtaiKey::to_ntt`] to convert a coefficient-domain key to NTT domain.
//!
//! # Decomposition Workflow (Hachi)
//!
//! For the Hachi decomposition-based workflow:
//!
//! ```text
//! t = A · G⁻¹(f₁, ..., fₗ) ∈ R_q^κ
//! ```
//!
//! Use [`commit_decomposed`] and [`verify_decomposed`]. These are
//! coefficient-domain only (type-enforced).
//!
//! # Security
//!
//! Binding relies on Module-SIS: finding short s₁ ≠ s₂ with A·s₁ = A·s₂
//! requires solving SIS in R_q^{κ × m}.

pub mod ajtai;

pub use ajtai::{commit_decomposed, verify_decomposed, AjtaiCommitment, AjtaiKey};
