//! Ajtai Commitment Benchmarks
//!
//! Benchmarks for Ajtai/SIS-based polynomial commitment, comparing
//! commitment times across different ring dimensions and NTT vs schoolbook.
//!
//! Reference: Hachi (https://eprint.iacr.org/2026/156)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use etale::commit::{commit_decomposed, AjtaiKey};
use etale::lattice::ntt::{RingElement, RingParams, COMPRESSED_K16, DILITHIUM_2, HACHI};
use etale::lattice::NttContext;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(42)
}

// ============================================================================
// Commitment Benchmarks (Schoolbook / Coefficient Domain)
// ============================================================================

/// Benchmark commitment with varying number of witnesses
fn bench_commit_witnesses(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_witnesses");
    let mut rng = bench_rng();

    let witness_counts = [1, 2, 4, 8];

    // Hachi params (d=1024)
    let (d, q, b, delta) = (HACHI.d, HACHI.q, HACHI.b, HACHI.delta);
    for &num_witnesses in &witness_counts {
        let key: AjtaiKey<RingElement> =
            AjtaiKey::random(&mut rng, HACHI.kappa, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        group.throughput(Throughput::Elements(num_witnesses as u64));
        group.bench_with_input(
            BenchmarkId::new("Hachi_d1024", num_witnesses),
            &(&key, &witnesses),
            |bench, (key, witnesses)| {
                bench.iter(|| commit_decomposed(black_box(key), black_box(witnesses), b, delta));
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
        let key: AjtaiKey<RingElement> =
            AjtaiKey::random(&mut rng, COMPRESSED_K16.kappa, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        group.throughput(Throughput::Elements(num_witnesses as u64));
        group.bench_with_input(
            BenchmarkId::new("Compressed_d64", num_witnesses),
            &(&key, &witnesses),
            |bench, (key, witnesses)| {
                bench.iter(|| commit_decomposed(black_box(key), black_box(witnesses), b, delta));
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

    let params: &[&RingParams] = &[&HACHI, &COMPRESSED_K16];

    for param in params {
        let key: AjtaiKey<RingElement> = AjtaiKey::random(
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
                bench.iter(|| commit_decomposed(black_box(key), black_box(witnesses), *b, *delta));
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

    for &kappa in &kappa_values {
        let key: AjtaiKey<RingElement> =
            AjtaiKey::random(&mut rng, kappa, delta * num_witnesses, d, q);
        let witnesses: Vec<_> = (0..num_witnesses)
            .map(|_| RingElement::random(&mut rng, d, q))
            .collect();

        let ring_muls = kappa * delta * num_witnesses;
        group.throughput(Throughput::Elements(ring_muls as u64));
        group.bench_with_input(
            BenchmarkId::new("kappa", kappa),
            &(&key, &witnesses),
            |bench, (key, witnesses)| {
                bench.iter(|| commit_decomposed(black_box(key), black_box(witnesses), b, delta));
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
                AjtaiKey::<RingElement>::random(
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
                AjtaiKey::<RingElement>::random(
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
// NTT vs Schoolbook Benchmarks
// ============================================================================

/// Compare coefficient domain (schoolbook) vs NTT domain commitment
fn bench_commit_ntt_vs_schoolbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_ntt_vs_schoolbook");
    let mut rng = bench_rng();

    // Dilithium params (NTT-friendly)
    let (d, q, psi) = (256, 8_380_417, 1753);
    let ctx = NttContext::new(d, q, psi);
    let num_cols = 8;

    // Coefficient domain
    let key_coeff: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 2, num_cols, d, q);
    let s_coeff: Vec<_> = (0..num_cols)
        .map(|_| RingElement::random(&mut rng, d, q))
        .collect();

    // NTT domain
    let key_ntt = key_coeff.to_ntt(&ctx);
    let s_ntt = ctx.forward_vec(&s_coeff);

    group.bench_function("schoolbook_d256", |bench| {
        bench.iter(|| key_coeff.commit(black_box(&s_coeff)));
    });

    group.bench_function("ntt_d256", |bench| {
        bench.iter(|| key_ntt.commit(black_box(&s_ntt)));
    });

    group.finish();
}

/// Benchmark raw matrix-vector product comparing schoolbook vs NTT
fn bench_raw_backend_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_raw_backend");
    let mut rng = bench_rng();

    // Dilithium params
    let (d, q, psi) = (DILITHIUM_2.d, DILITHIUM_2.q, 1753);
    let ctx = NttContext::new(d, q, psi);
    let cols = 8;

    let key_coeff: AjtaiKey<RingElement> = AjtaiKey::random(&mut rng, 2, cols, d, q);
    let s_coeff: Vec<_> = (0..cols)
        .map(|_| RingElement::random(&mut rng, d, q))
        .collect();

    let key_ntt = key_coeff.to_ntt(&ctx);
    let s_ntt = ctx.forward_vec(&s_coeff);

    group.throughput(Throughput::Elements((2 * cols) as u64)); // κ × m ring muls

    group.bench_with_input(
        BenchmarkId::new("schoolbook", "2x8"),
        &(&key_coeff, &s_coeff),
        |bench, (key, s)| {
            bench.iter(|| key.commit(black_box(s)));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ntt", "2x8"),
        &(&key_ntt, &s_ntt),
        |bench, (key, s)| {
            bench.iter(|| key.commit(black_box(s)));
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
    bench_commit_ntt_vs_schoolbook,
    bench_raw_backend_comparison,
);
criterion_main!(benches);
