//! Demonstration of Fiat-Shamir with Aborts
//!
//! This shows the key concepts from Lyubashevsky's 2009 paper.

use fiat_shamir_aborts::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() {
    println!("=== Fiat-Shamir with Aborts Demo ===\n");
    println!("Based on Lyubashevsky 2009: 'Fiat-Shamir with Aborts:");
    println!("Applications to Lattice and Factoring-Based Signatures'\n");

    let mut rng = ChaCha20Rng::seed_from_u64(42);

    // Demo 1: Basic identification protocol
    demo_identification(&mut rng);

    // Demo 2: Signature scheme
    demo_signature(&mut rng);

    // Demo 3: Rejection sampling analysis
    demo_rejection_sampling(&mut rng);
}

fn demo_identification(rng: &mut ChaCha20Rng) {
    println!("--- Demo 1: Identification Protocol ---\n");

    let params = IdentificationSchemeParameters::default();
    println!("Parameters:");
    println!("  Dimension n = {}", params.n);
    println!("  Modulus q = {}", params.q);
    println!("  Randomness bound B = {}", params.b);
    println!("  Challenge bound = {}", params.challenge_bound);
    println!("  Rejection bound = {}\n", params.rejection_bound());

    let mut scheme = IdentificationScheme::new(params);

    // Key generation
    let keypair = scheme.keygen(rng);
    println!("Generated keypair:");
    println!("  Secret key s = {}", keypair.sk.s);
    println!("  ||s||_∞ = {}\n", keypair.sk.s.ell_inf_norm());

    // Run identification
    println!("Running identification protocol...");
    match scheme.identify(rng, &keypair, 100) {
        Some((commitment, challenge, z)) => {
            println!("  Commitment w = {}", commitment.w);
            println!("  Challenge c = {}", challenge.c);
            println!("  Response z = {}", z);
            println!("  ||z||_∞ = {}", z.ell_inf_norm());

            // Verify
            let valid = scheme.verify(&keypair.pk, &commitment, &challenge, &z);
            println!(
                "\n  Verification: {}",
                if valid { "PASS ✓" } else { "FAIL ✗" }
            );
        }
        None => {
            println!("  Identification failed after max attempts");
        }
    }

    println!("\nStatistics:");
    println!("  Total attempts: {}", scheme.stats.total_attempts);
    println!("  Aborts: {}", scheme.stats.aborts);
    println!("  Successes: {}", scheme.stats.successes);
    println!("  Abort rate: {:.1}%\n", scheme.stats.abort_rate() * 100.0);
}

fn demo_signature(rng: &mut ChaCha20Rng) {
    println!("--- Demo 2: Signature Scheme (Fiat-Shamir Transform) ---\n");

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
    println!("Message: {:?}\n", String::from_utf8_lossy(message));

    // Sign
    println!("Signing...");
    match scheme.sign(rng, &keypair, message, 100) {
        Some(sig) => {
            println!("  Signature commitment w = {}", sig.w);
            println!("  Signature response z = {}", sig.z);
            println!("  ||z||_∞ = {}", sig.z.ell_inf_norm());

            // Verify
            let valid = scheme.verify(&keypair.pk, message, &sig);
            println!(
                "\n  Verification: {}",
                if valid { "PASS ✓" } else { "FAIL ✗" }
            );

            // Try wrong message
            let wrong = b"Tampered message";
            let invalid = scheme.verify(&keypair.pk, wrong, &sig);
            println!(
                "  Wrong message verification: {}",
                if invalid {
                    "PASS (BAD!)"
                } else {
                    "FAIL ✓ (as expected)"
                }
            );
        }
        None => {
            println!("  Signing failed");
        }
    }

    println!("\nStatistics:");
    println!(
        "  Abort rate: {:.1}%\n",
        scheme.stats().abort_rate() * 100.0
    );
}

fn demo_rejection_sampling(rng: &mut ChaCha20Rng) {
    println!("--- Demo 3: Rejection Sampling Analysis ---\n");

    println!("The KEY insight of Lyubashevsky 2009:");
    println!("Without rejection sampling, z = y + sc leaks information about s.");
    println!("With rejection sampling, z is uniform over a range independent of s.\n");

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

    println!("Abort rates for different parameter choices:\n");
    println!(
        "{:<20} {:>10} {:>15} {:>12}",
        "Config", "B", "Rej. Bound", "Abort Rate"
    );
    println!("{}", "-".repeat(60));

    for (name, params) in test_configs {
        let rej_bound = params.rejection_bound();
        let b = params.b;
        let mut scheme = IdentificationScheme::new(params);
        let keypair = scheme.keygen(rng);

        // Run many identifications
        for _ in 0..200 {
            scheme.identify(rng, &keypair, 50);
        }

        println!(
            "{:<20} {:>10} {:>15} {:>11.1}%",
            name,
            b,
            rej_bound,
            scheme.stats.abort_rate() * 100.0
        );
    }

    println!("\n");
    println!("Theoretical insight:");
    println!("- Larger B → larger safe range → fewer aborts");
    println!("- But larger B → larger signatures");
    println!("- The 2009 paper's contribution: aborts DON'T go into the signature!");
    println!("  (Unlike earlier schemes where failed attempts increased size)");
}
