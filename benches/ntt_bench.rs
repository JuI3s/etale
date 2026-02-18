//! NTT and Ring Arithmetic Benchmarks
//!
//! Micro-benchmarks for ring operations across different parameter regimes,
//! designed to evaluate trace compression performance for lattice-based
//! polynomial commitments.
//!
//! Reference: Hachi (https://eprint.iacr.org/2026/156)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use etale::lattice::ntt::{
    compute_tower_coset_reps, decompose_poly, enumerate_subgroup, recompose_poly, sparse_mul,
    trace_naive, trace_tower, NttTables, RingElement, RingParams, SparseChallenge,
    COMPRESSED_K16, COMPRESSED_K32, COMPRESSED_K4, COMPRESSED_K8, DILITHIUM_2, FALCON_512,
    GREYHOUND, HACHI,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(42)
}

// ============================================================================
// Ring Multiplication Benchmarks
// ============================================================================

/// Benchmark schoolbook multiplication across parameter sets
fn bench_ring_mul_schoolbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_mul_schoolbook");
    let mut rng = bench_rng();

    let params: &[&RingParams] = &[
        &HACHI,
        &COMPRESSED_K4,
        &COMPRESSED_K8,
        &COMPRESSED_K16,
        &COMPRESSED_K32,
        &GREYHOUND,
    ];

    for param in params {
        let a = RingElement::random(&mut rng, param.d, param.q);
        let b = RingElement::random(&mut rng, param.d, param.q);

        group.throughput(Throughput::Elements(param.d as u64));
        group.bench_with_input(
            BenchmarkId::new(param.name, format!("d={}", param.d)),
            &(a, b),
            |bench, (a, b)| bench.iter(|| a.mul_schoolbook(black_box(b))),
        );
    }

    group.finish();
}

/// Benchmark NTT-based multiplication for NTT-friendly primes
fn bench_ring_mul_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_mul_ntt");
    let mut rng = bench_rng();

    // Only properly NTT-friendly parameter sets (q ≡ 1 mod 2d)
    // Note: Kyber uses q ≡ 1 mod 256 (not 512), requiring a modified NTT
    let ntt_params: &[(&RingParams, u64)] = &[
        (&DILITHIUM_2, 1753),  // psi for Dilithium (q=8380417, d=256)
        (&FALCON_512, 49),     // psi for Falcon (q=12289, d=512)
    ];

    for &(param, psi) in ntt_params {
        let tables = NttTables::new(param.d, param.q, psi);
        let a = RingElement::random(&mut rng, param.d, param.q);
        let b = RingElement::random(&mut rng, param.d, param.q);

        group.throughput(Throughput::Elements(param.d as u64));
        group.bench_with_input(
            BenchmarkId::new(param.name, format!("d={}", param.d)),
            &(tables, a, b),
            |bench, (tables, a, b)| bench.iter(|| tables.ring_mul(black_box(a), black_box(b))),
        );
    }

    group.finish();
}

/// Benchmark multiplication with small operand (decomposed, ≤4 bits)
fn bench_ring_mul_small_operand(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_mul_small_operand");
    let mut rng = bench_rng();

    let dims = [32, 64, 128, 256, 512, 1024];
    let q = HACHI.q;

    for &d in &dims {
        // Full-size operand
        let a = RingElement::random(&mut rng, d, q);
        // Small operand (coefficients in {0,...,15})
        let b = RingElement::random_bounded(&mut rng, d, q, 16);

        group.throughput(Throughput::Elements(d as u64));
        group.bench_with_input(
            BenchmarkId::new("schoolbook", format!("d={}", d)),
            &(a, b),
            |bench, (a, b)| bench.iter(|| a.mul_schoolbook(black_box(b))),
        );
    }

    group.finish();
}

// ============================================================================
// Sparse Multiplication Benchmarks
// ============================================================================

/// Benchmark sparse multiplication (challenge × polynomial)
fn bench_sparse_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_mul");
    let mut rng = bench_rng();

    // Hachi uses c=16 nonzero coefficients
    let nonzero_counts = [8, 16, 32, 64];
    let dims = [64, 256, 512, 1024];
    let q = HACHI.q;

    for &d in &dims {
        for &c_nonzero in &nonzero_counts {
            if c_nonzero > d {
                continue;
            }

            let f = RingElement::random(&mut rng, d, q);
            let g = SparseChallenge::random(&mut rng, d, c_nonzero);

            group.throughput(Throughput::Elements((c_nonzero * d) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("d={}", d), format!("c={}", c_nonzero)),
                &(f, g),
                |bench, (f, g)| bench.iter(|| sparse_mul(black_box(f), black_box(g))),
            );
        }
    }

    group.finish();
}

/// Compare sparse vs schoolbook for challenge multiplication
fn bench_sparse_vs_schoolbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_schoolbook");
    let mut rng = bench_rng();

    // Hachi parameters: d=1024, c=16
    let d = 1024;
    let c_nonzero = 16;
    let q = HACHI.q;

    let f = RingElement::random(&mut rng, d, q);
    let g_sparse = SparseChallenge::random(&mut rng, d, c_nonzero);
    let g_dense = g_sparse.to_ring_element(q);

    group.bench_function("sparse_c16", |bench| {
        bench.iter(|| sparse_mul(black_box(&f), black_box(&g_sparse)))
    });

    group.bench_function("schoolbook", |bench| {
        bench.iter(|| f.mul_schoolbook(black_box(&g_dense)))
    });

    group.finish();
}

// ============================================================================
// NTT Transform Benchmarks
// ============================================================================

/// Benchmark forward NTT transform
fn bench_ntt_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_forward");
    let mut rng = bench_rng();

    // Only properly NTT-friendly parameter sets
    let ntt_params: &[(&RingParams, u64)] = &[
        (&DILITHIUM_2, 1753),
        (&FALCON_512, 49),
    ];

    for &(param, psi) in ntt_params {
        let tables = NttTables::new(param.d, param.q, psi);
        let a = RingElement::random(&mut rng, param.d, param.q);

        group.throughput(Throughput::Elements(param.d as u64));
        group.bench_with_input(
            BenchmarkId::new(param.name, format!("d={}", param.d)),
            &(tables, a),
            |bench, (tables, a)| {
                bench.iter_batched(
                    || a.coeffs.clone(),
                    |mut coeffs| {
                        tables.forward(black_box(&mut coeffs));
                        coeffs
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark inverse NTT transform
fn bench_ntt_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_inverse");
    let mut rng = bench_rng();

    // Only properly NTT-friendly parameter sets
    let ntt_params: &[(&RingParams, u64)] = &[
        (&DILITHIUM_2, 1753),
        (&FALCON_512, 49),
    ];

    for &(param, psi) in ntt_params {
        let tables = NttTables::new(param.d, param.q, psi);
        let a = RingElement::random(&mut rng, param.d, param.q);

        // Start from NTT form
        let mut a_ntt = a.coeffs.clone();
        tables.forward(&mut a_ntt);

        group.throughput(Throughput::Elements(param.d as u64));
        group.bench_with_input(
            BenchmarkId::new(param.name, format!("d={}", param.d)),
            &(tables, a_ntt),
            |bench, (tables, a_ntt)| {
                bench.iter_batched(
                    || a_ntt.clone(),
                    |mut coeffs| {
                        tables.inverse(black_box(&mut coeffs));
                        coeffs
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// ============================================================================
// Automorphism Benchmarks
// ============================================================================

/// Benchmark automorphism σ_k
fn bench_automorphism(c: &mut Criterion) {
    let mut group = c.benchmark_group("automorphism");
    let mut rng = bench_rng();

    let dims = [64, 128, 256, 512, 1024];
    let q = HACHI.q;

    // Common automorphisms used in trace computation
    let automorphisms = [
        (3, "σ_3"),
        (17, "σ_17"),
        (65, "σ_65"),
    ];

    for &d in &dims {
        let a = RingElement::random(&mut rng, d, q);

        for &(k, name) in &automorphisms {
            let k = k % (2 * d); // Ensure valid automorphism

            group.throughput(Throughput::Elements(d as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("d={}", d), name),
                &(a.clone(), k),
                |bench, (a, k)| bench.iter(|| a.automorphism(black_box(*k))),
            );
        }
    }

    group.finish();
}

// ============================================================================
// Trace Computation Benchmarks
// ============================================================================

/// Benchmark tower-optimized trace vs naive
fn bench_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace");
    let mut rng = bench_rng();

    let q = HACHI.q;
    let d = 1024;

    // Different compression factors
    let compression_factors = [4, 8, 16, 32];

    for &k in &compression_factors {
        let coset_reps = compute_tower_coset_reps(d, k);
        let subgroup = enumerate_subgroup(d, k);
        let x = RingElement::random(&mut rng, d, q);

        group.bench_with_input(
            BenchmarkId::new("tower", format!("k={}", k)),
            &(x.clone(), coset_reps.clone()),
            |bench, (x, reps)| bench.iter(|| trace_tower(black_box(x), black_box(reps))),
        );

        // Only benchmark naive for small subgroups (too slow otherwise)
        if k <= 16 {
            group.bench_with_input(
                BenchmarkId::new("naive", format!("k={}", k)),
                &(x, subgroup),
                |bench, (x, subgroup)| bench.iter(|| trace_naive(black_box(x), black_box(subgroup))),
            );
        }
    }

    group.finish();
}

/// Benchmark trace at Hachi parameters (d=1024, k=4 → |H|=256)
fn bench_trace_hachi_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_hachi");
    let mut rng = bench_rng();

    // Hachi: d=1024, extension field k_ext=4
    // Compression to R_q^H where |H| = d/k_ext = 256
    let d = 1024;
    let k_ext = 4;
    let compression_factor = d / k_ext; // 256
    let q = HACHI.q;

    let coset_reps = compute_tower_coset_reps(d, compression_factor);
    let x = RingElement::random(&mut rng, d, q);

    // Tower: log_2(256) = 8 automorphisms
    group.bench_function("tower_8_levels", |bench| {
        bench.iter(|| trace_tower(black_box(&x), black_box(&coset_reps)))
    });

    // Report theoretical speedup
    let naive_ops = compression_factor;
    let tower_ops = (compression_factor as f64).log2() as usize;
    println!(
        "Theoretical speedup: {}× ({} automorphisms → {})",
        naive_ops / tower_ops,
        naive_ops,
        tower_ops
    );

    group.finish();
}

// ============================================================================
// Decomposition Benchmarks
// ============================================================================

/// Benchmark base-b decomposition
fn bench_decompose(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompose");
    let mut rng = bench_rng();

    // Hachi: b=16, δ=8
    let decomp_params: &[(&str, usize, u64, usize)] = &[
        ("Hachi_b16", 1024, 16, 8),
        ("Greyhound_b256", 64, 256, 4),
        ("Compressed_k16", 64, 16, 8),
    ];

    for &(name, d, b, delta) in decomp_params {
        let q = HACHI.q;
        let f = RingElement::random(&mut rng, d, q);

        group.throughput(Throughput::Elements(d as u64));
        group.bench_with_input(
            BenchmarkId::new("decompose", name),
            &(f, b, delta),
            |bench, (f, b, delta)| bench.iter(|| decompose_poly(black_box(f), *b, *delta)),
        );
    }

    group.finish();
}

/// Benchmark recomposition from base-b digits
fn bench_recompose(c: &mut Criterion) {
    let mut group = c.benchmark_group("recompose");
    let mut rng = bench_rng();

    let decomp_params: &[(&str, usize, u64, usize)] = &[
        ("Hachi_b16", 1024, 16, 8),
        ("Greyhound_b256", 64, 256, 4),
    ];

    for &(name, d, b, delta) in decomp_params {
        let q = HACHI.q;
        let f = RingElement::random(&mut rng, d, q);
        let decomposed = decompose_poly(&f, b, delta);

        group.throughput(Throughput::Elements((d * delta) as u64));
        group.bench_with_input(
            BenchmarkId::new("recompose", name),
            &(decomposed, b),
            |bench, (decomposed, b)| bench.iter(|| recompose_poly(black_box(decomposed), *b)),
        );
    }

    group.finish();
}

// ============================================================================
// Comparison Benchmarks
// ============================================================================

/// Compare ring multiplication costs across parameter regimes
fn bench_ring_mul_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_mul_comparison");
    group.sample_size(100);

    let mut rng = bench_rng();

    // Same modulus, different ring dimensions
    let q = HACHI.q;
    let dims = [32, 64, 128, 256, 512, 1024];

    for &d in &dims {
        let a = RingElement::random(&mut rng, d, q);
        let b = RingElement::random(&mut rng, d, q);

        group.bench_with_input(
            BenchmarkId::new("schoolbook", format!("d={}", d)),
            &(a, b),
            |bench, (a, b)| bench.iter(|| a.mul_schoolbook(black_box(b))),
        );
    }

    // Compare with NTT-friendly primes
    let kyber_tables = NttTables::new(256, 3329, 17);
    let a_kyber = RingElement::random(&mut rng, 256, 3329);
    let b_kyber = RingElement::random(&mut rng, 256, 3329);

    group.bench_with_input(
        BenchmarkId::new("ntt", "d=256_Kyber"),
        &(kyber_tables, a_kyber, b_kyber),
        |bench, (tables, a, b)| bench.iter(|| tables.ring_mul(black_box(a), black_box(b))),
    );

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    name = ring_arithmetic;
    config = Criterion::default();
    targets = bench_ring_mul_schoolbook,
              bench_ring_mul_ntt,
              bench_ring_mul_small_operand,
              bench_ring_mul_comparison
);

criterion_group!(
    name = sparse_ops;
    config = Criterion::default();
    targets = bench_sparse_mul,
              bench_sparse_vs_schoolbook
);

criterion_group!(
    name = ntt_transforms;
    config = Criterion::default();
    targets = bench_ntt_forward,
              bench_ntt_inverse
);

criterion_group!(
    name = automorphisms;
    config = Criterion::default();
    targets = bench_automorphism
);

criterion_group!(
    name = trace_ops;
    config = Criterion::default();
    targets = bench_trace,
              bench_trace_hachi_params
);

criterion_group!(
    name = decomposition;
    config = Criterion::default();
    targets = bench_decompose,
              bench_recompose
);

criterion_main!(
    ring_arithmetic,
    sparse_ops,
    ntt_transforms,
    automorphisms,
    trace_ops,
    decomposition
);

