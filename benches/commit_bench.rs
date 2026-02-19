//! Ajtai Commitment Benchmarks
//!
//! Benchmarks for Ajtai/SIS-based polynomial commitment, comparing
//! commitment times across different ring dimensions and multiplication backends.
//!
//! Reference: Hachi (https://eprint.iacr.org/2026/156)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use etale::commit::{commit_decomposed, AjtaiKey};
use etale::lattice::ntt::{
    NegacyclicNtt, RingElement, RingParams, COMPRESSED_K16, DILITHIUM_2, HACHI,
};
use etale::lattice::{NttBackend, SchoolbookBackend};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(42)
}

// ============================================================================
// Commitment Benchmarks (Schoolbook)
// ============================================================================

/// Benchmark commitment with varying number of witnesses
fn bench_commit_witnesses(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_witnesses");
    let mut rng = bench_rng();

    let witness_counts = [1, 2, 4, 8];
    let backend = SchoolbookBackend;

    // Hachi params (d=1024)
    let (d, q, b, delta) = (HACHI.d, HACHI.q, HACHI.b, HACHI.delta);
    for &num_witnesses in &witness_counts {
        let key = AjtaiKey::random(&mut rng, HACHI.kappa, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        group.throughput(Throughput::Elements(num_witnesses as u64));
        group.bench_with_input(
            BenchmarkId::new("Hachi_d1024", num_witnesses),
            &(&key, &witnesses),
            |bench, (key, witnesses)| {
                bench.iter(|| {
                    commit_decomposed(black_box(key), black_box(witnesses), b, delta, &backend)
                });
            },
        );
    }

    // Compressed k=16 params (d=64)
    let (d, q, b, delta) = (
        COMPRESSED_K16.d,
        COMPRESSED_K16.q,
        COMPRESSED_K16.b,
        COMPRESSED_K16.delta,
    );
    for &num_witnesses in &witness_counts {
        let key = AjtaiKey::random(&mut rng, COMPRESSED_K16.kappa, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        group.throughput(Throughput::Elements(num_witnesses as u64));
        group.bench_with_input(
            BenchmarkId::new("Compressed_d64", num_witnesses),
            &(&key, &witnesses),
            |bench, (key, witnesses)| {
                bench.iter(|| {
                    commit_decomposed(black_box(key), black_box(witnesses), b, delta, &backend)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark commitment comparing different ring dimensions
fn bench_commit_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_dimensions");
    let mut rng = bench_rng();

    let num_witnesses = 4;
    let backend = SchoolbookBackend;

    let params: &[&RingParams] = &[&HACHI, &COMPRESSED_K16];

    for param in params {
        let key = AjtaiKey::random(
            &mut rng,
            param.kappa,
            param.delta * num_witnesses,
            param.d,
            param.q,
        );
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, param.d, param.q))
            .collect();

        group.throughput(Throughput::Elements((param.d * num_witnesses) as u64));
        group.bench_with_input(
            BenchmarkId::new(param.name, format!("d={}", param.d)),
            &(&key, &witnesses, param.b, param.delta),
            |bench, (key, witnesses, b, delta)| {
                bench.iter(|| {
                    commit_decomposed(black_box(key), black_box(witnesses), *b, *delta, &backend)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark matrix-vector product with varying kappa
fn bench_matvec_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_matvec");
    let mut rng = bench_rng();

    let kappa_values = [1, 2, 4];
    let num_witnesses = 1;
    let (d, q, b, delta) = (64, HACHI.q, 16u64, 8usize);
    let backend = SchoolbookBackend;

    for &kappa in &kappa_values {
        let key = AjtaiKey::random(&mut rng, kappa, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        let ring_muls = kappa * delta * num_witnesses;
        group.throughput(Throughput::Elements(ring_muls as u64));
        group.bench_with_input(
            BenchmarkId::new("kappa", kappa),
            &(&key, &witnesses),
            |bench, (key, witnesses)| {
                bench.iter(|| {
                    commit_decomposed(black_box(key), black_box(witnesses), b, delta, &backend)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark key generation
fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_keygen");

    let num_witnesses = 4;

    // Hachi params
    let (d, q, delta) = (HACHI.d, HACHI.q, HACHI.delta);
    group.bench_function("Hachi_d1024", |bench| {
        bench.iter_batched(
            bench_rng,
            |mut rng| {
                AjtaiKey::random(
                    &mut rng,
                    HACHI.kappa,
                    black_box(delta * num_witnesses),
                    black_box(d),
                    black_box(q),
                )
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Compressed params
    let (d, q, delta) = (COMPRESSED_K16.d, COMPRESSED_K16.q, COMPRESSED_K16.delta);
    group.bench_function("Compressed_d64", |bench| {
        bench.iter_batched(
            bench_rng,
            |mut rng| {
                AjtaiKey::random(
                    &mut rng,
                    COMPRESSED_K16.kappa,
                    black_box(delta * num_witnesses),
                    black_box(d),
                    black_box(q),
                )
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ============================================================================
// Backend Comparison Benchmarks (Schoolbook vs NTT)
// ============================================================================

/// Compare schoolbook vs NTT backends for NTT-friendly parameter sets
fn bench_backend_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_backend");
    let mut rng = bench_rng();

    // Dilithium params (NTT-friendly: q = 8380417 ≡ 1 mod 512)
    let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);
    let (b, delta) = (DILITHIUM_2.b, DILITHIUM_2.delta);
    let num_witnesses = 2;

    let key = AjtaiKey::random(&mut rng, DILITHIUM_2.kappa, delta * num_witnesses, d, q);
    let witnesses: Vec<_> = (0..num_witnesses)
        .map(|_| RingElement::random(&mut rng, d, q))
        .collect();

    let schoolbook = SchoolbookBackend;
    let ntt = NttBackend::new(NegacyclicNtt::new(d, q, psi));

    group.throughput(Throughput::Elements(num_witnesses as u64));

    group.bench_with_input(
        BenchmarkId::new("Dilithium_schoolbook", "d=256"),
        &(&key, &witnesses),
        |bench, (key, witnesses)| {
            bench.iter(|| {
                commit_decomposed(black_box(key), black_box(witnesses), b, delta, &schoolbook)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Dilithium_ntt", "d=256"),
        &(&key, &witnesses),
        |bench, (key, witnesses)| {
            bench.iter(|| commit_decomposed(black_box(key), black_box(witnesses), b, delta, &ntt));
        },
    );

    group.finish();
}

/// Benchmark raw matrix-vector product with different backends
fn bench_raw_backend_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_raw_backend");
    let mut rng = bench_rng();

    // Dilithium params
    let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);
    let cols = 8;

    let key = AjtaiKey::random(&mut rng, 2, cols, d, q);
    let s: Vec<_> = (0..cols)
        .map(|_| RingElement::random(&mut rng, d, q))
        .collect();

    let schoolbook = SchoolbookBackend;
    let ntt = NttBackend::new(NegacyclicNtt::new(d, q, psi));

    group.throughput(Throughput::Elements((2 * cols) as u64)); // κ × m ring muls

    group.bench_with_input(
        BenchmarkId::new("schoolbook", "2x8"),
        &(&key, &s),
        |bench, (key, s)| {
            bench.iter(|| key.commit_with(black_box(s), &schoolbook));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ntt", "2x8"),
        &(&key, &s),
        |bench, (key, s)| {
            bench.iter(|| key.commit_with(black_box(s), &ntt));
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_commit_witnesses,
    bench_commit_dimensions,
    bench_matvec_product,
    bench_keygen,
    bench_backend_comparison,
    bench_raw_backend_comparison,
);
criterion_main!(benches);
