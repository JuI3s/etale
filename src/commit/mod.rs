//! Lattice-based polynomial commitment schemes
//!
//! This module provides Ajtai/SIS-based commitment schemes for polynomial vectors.
//!
//! # Core Interface
//!
//! [`AjtaiKey`] is a pure random matrix A ∈ R_q^{κ × m}. The commitment to a
//! short vector s is simply `t = A · s` via [`AjtaiKey::commit`].
//!
//! # Decomposition Workflow (Hachi)
//!
//! For the Hachi decomposition-based workflow:
//!
//! ```text
//! t = A · G⁻¹(f₁, ..., fₗ) ∈ R_q^κ
//! ```
//!
//! Use [`commit_decomposed`] and [`verify_decomposed`].
//!
//! # Security
//!
//! Binding relies on Module-SIS: finding short s₁ ≠ s₂ with A·s₁ = A·s₂
//! requires solving SIS in R_q^{κ × m}.

pub mod ajtai;

pub use ajtai::{commit_decomposed, verify_decomposed, AjtaiCommitment, AjtaiKey};
