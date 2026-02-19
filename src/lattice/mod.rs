//! Core lattice operations and types
//!
//! This module provides fundamental data structures for working with lattices,
//! including vectors and matrices over Z_q.

pub mod decompose;
pub mod matrix;
pub mod modular;
pub mod ntt;
pub mod pairing;
pub mod params;
pub mod ring;
pub mod ring_mul;
pub mod splitting;
pub mod trace;
pub mod vector;

pub use matrix::Matrix;
pub use modular::{add_mod, mod_inv, mul_mod, neg_mod, pow_mod, sub_mod, Zq};
pub use params::{
    find_primitive_2d_root, RingParams, ALL_PARAMS, COMPRESSED_K16, COMPRESSED_K32, COMPRESSED_K4,
    COMPRESSED_K8, DILITHIUM_2, FALCON_512, GREYHOUND, HACHI, HACHI_FAMILY, KYBER_512,
    NTT_FRIENDLY,
};
pub use ring_mul::{NttBackend, RingMulBackend, SchoolbookBackend};
pub use splitting::{Challenge, ChallengeSet, SplittingParams};
pub use vector::Vector;
