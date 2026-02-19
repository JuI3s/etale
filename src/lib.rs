//! # Étale
//!
//! A general-purpose library for lattice-based cryptography, including:
//!
//! - **Core lattice operations**: Vectors and matrices over Z_q
//! - **Zero-knowledge support**: Rejection sampling and Fiat-Shamir schemes
//! - **Polynomial commitments**: Lattice-based polynomial commitment schemes (coming soon)
//!
//! ## Modules
//!
//! - [`lattice`]: Core lattice operations (vectors, matrices)
//! - [`zk`]: Zero-knowledge proof support (rejection sampling, identification schemes)
//! - [`commit`]: Polynomial commitment schemes (Ajtai/SIS-based)

pub mod commit;
pub mod lattice;
pub mod zk;

// Re-export commonly used types
pub use lattice::{Matrix, Vector};
pub use zk::{
    Challenge, Commitment, IdentificationScheme, IdentificationSchemeParameters, KeyPair,
    ProverState, PublicKey, RejectionSamplingParams, RejectionSamplingStats, Response, SecretKey,
    Signature, SignatureScheme,
};
