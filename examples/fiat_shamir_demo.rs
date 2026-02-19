//! Demonstration of Fiat-Shamir with Aborts
//!
//! This shows the key concepts from Lyubashevsky's 2009 paper.
//!
//! Run with `RUST_LOG=info cargo run --example fiat_shamir_demo` to see output.
//! For more verbose output, use `RUST_LOG=debug`.

use etale::{IdentificationScheme, IdentificationSchemeParameters, SignatureScheme};
use log::info;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("=== Fiat-Shamir with Aborts Demo ===");
    info!("Based on Lyubashevsky 2009: 'Fiat-Shamir with Aborts:");
    info!("Applications to Lattice and Factoring-Based Signatures'");
    info!("");

    let mut rng = ChaCha20Rng::seed_from_u64(42);

    // Demo 1: Basic identification protocol
    demo_identification(&mut rng);

    // Demo 2: Signature scheme
    demo_signature(&mut rng);

    // Demo 3: Rejection sampling analysis
    demo_rejection_sampling(&mut rng);
}

fn demo_identification(rng: &mut ChaCha20Rng) {
    info!("--- Demo 1: Identification Protocol ---");
    info!("");

    let params = IdentificationSchemeParameters::default();
    info!("Parameters:");
    info!("  Dimension n = {}", params.n);
    info!("  Modulus q = {}", params.q);
    info!("  Randomness bound B = {}", params.b);
    info!("  Challenge bound = {}", params.challenge_bound);
    info!("  Rejection bound = {}", params.rejection_bound());
    info!("");

    let mut scheme = IdentificationScheme::new(params);

    // Key generation
    let keypair = scheme.keygen(rng);
    info!("Generated keypair:");
    info!("  Secret key s = {}", keypair.sk.s);
    info!("  ||s||_∞ = {}", keypair.sk.s.ell_inf_norm());
    info!("");

    // Run identification
    info!("Running identification protocol...");
    match scheme.prove(rng, &keypair, 100) {
        Some((commitment, challenge, z)) => {
            info!("  Commitment w = {}", commitment.w);
            info!("  Challenge c = {}", challenge.c);
            info!("  Response z = {z}");
            info!("  ||z||_∞ = {}", z.ell_inf_norm());

            // Verify
            let valid = scheme.verify(&keypair.pk, &commitment, &challenge, &z);
            info!("");
            info!(
                "  Verification: {}",
                if valid { "PASS ✓" } else { "FAIL ✗" }
            );
        }
        None => {
            info!("  Identification failed after max attempts");
        }
    }

    info!("");
    info!("Statistics:");
    info!("  Total attempts: {}", scheme.stats.total_attempts);
    info!("  Aborts: {}", scheme.stats.aborts);
    info!("  Successes: {}", scheme.stats.successes);
    info!("  Abort rate: {:.1}%", scheme.stats.abort_rate() * 100.0);
    info!("");
}

fn demo_signature(rng: &mut ChaCha20Rng) {
    info!("--- Demo 2: Signature Scheme (Fiat-Shamir Transform) ---");
    info!("");

    let params = IdentificationSchemeParameters {
        n: 4,
        q: 7681,
        b: 2000,
        challenge_bound: 1,
        secret_bound: 1,
    };

    let mut scheme = SignatureScheme::new(params);
    let keypair = scheme.keygen(rng);

    let message = b"Lattice-based cryptography is post-quantum secure!";
    info!("Message: {:?}", String::from_utf8_lossy(message));
    info!("");

    // Sign
    info!("Signing...");
    match scheme.sign(rng, &keypair, message, 100) {
        Some(sig) => {
            info!("  Signature commitment w = {}", sig.w);
            info!("  Signature response z = {}", sig.z);
            info!("  ||z||_∞ = {}", sig.z.ell_inf_norm());

            // Verify
            let valid = scheme.verify(&keypair.pk, message, &sig);
            info!("");
            info!(
                "  Verification: {}",
                if valid { "PASS ✓" } else { "FAIL ✗" }
            );

            // Try wrong message
            let wrong = b"Tampered message";
            let invalid = scheme.verify(&keypair.pk, wrong, &sig);
            info!(
                "  Wrong message verification: {}",
                if invalid {
                    "PASS (BAD!)"
                } else {
                    "FAIL ✓ (as expected)"
                }
            );
        }
        None => {
            info!("  Signing failed");
        }
    }

    info!("");
    info!("Statistics:");
    info!("  Abort rate: {:.1}%", scheme.stats().abort_rate() * 100.0);
    info!("");
}

fn demo_rejection_sampling(rng: &mut ChaCha20Rng) {
    info!("--- Demo 3: Rejection Sampling Analysis ---");
    info!("");

    info!("The KEY insight of Lyubashevsky 2009:");
    info!("Without rejection sampling, z = y + sc leaks information about s.");
    info!("With rejection sampling, z is uniform over a range independent of s.");
    info!("");

    // Compare abort rates with different parameters
    let test_configs = vec![
        (
            "Tight (high abort)",
            IdentificationSchemeParameters {
                n: 4,
                q: 7681,
                b: 100,
                challenge_bound: 1,
                secret_bound: 1,
            },
        ),
        (
            "Medium",
            IdentificationSchemeParameters {
                n: 4,
                q: 7681,
                b: 500,
                challenge_bound: 1,
                secret_bound: 1,
            },
        ),
        (
            "Loose (low abort)",
            IdentificationSchemeParameters {
                n: 4,
                q: 7681,
                b: 2000,
                challenge_bound: 1,
                secret_bound: 1,
            },
        ),
    ];

    info!("Abort rates for different parameter choices:");
    info!("");
    info!(
        "{:<20} {:>10} {:>15} {:>12}",
        "Config", "B", "Rej. Bound", "Abort Rate"
    );
    info!("{}", "-".repeat(60));

    for (name, params) in test_configs {
        let rej_bound = params.rejection_bound();
        let b = params.b;
        let mut scheme = IdentificationScheme::new(params);
        let keypair = scheme.keygen(rng);

        // Run many identifications
        for _ in 0..200 {
            scheme.prove(rng, &keypair, 50);
        }

        info!(
            "{:<20} {:>10} {:>15} {:>11.1}%",
            name,
            b,
            rej_bound,
            scheme.stats.abort_rate() * 100.0
        );
    }

    info!("");
    info!("Theoretical insight:");
    info!("- Larger B → larger safe range → fewer aborts");
    info!("- But larger B → larger signatures");
    info!("- The 2009 paper's contribution: aborts DON'T go into the signature!");
    info!("  (Unlike earlier schemes where failed attempts increased size)");
}
