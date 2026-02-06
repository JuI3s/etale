#![allow(non_local_definitions)]

use ark_ff::fields::models::fp::{Fp64, MontBackend, MontConfig};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use etale::lattice::trace::{
    trace_naive, trace_tower, CyclotomicRingElement, GaloisSubgroup,
};
use rand::thread_rng;

// Test field configuration for benchmarking
#[derive(MontConfig)]
#[modulus = "65557"]
#[generator = "2"]
#[small_subgroup_base = "0"]
#[small_subgroup_power = "0"]
struct BenchFqConfig;

type BenchFq = Fp64<MontBackend<BenchFqConfig, 1>>;

// Benchmark parameters
const TRACE_PARAMS: &[(usize, usize)] = &[(64, 4), (128, 4), (256, 4), (512, 4), (1024, 4)];
const AUTOMORPHISM_DIMS: &[usize] = &[64, 128, 256, 512, 1024];
const AUTOMORPHISMS: &[usize] = &[1, 17, 33, 65]; // Common automorphisms

fn bench_trace_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_naive");

    for &(d, k) in TRACE_PARAMS {
        let h = GaloisSubgroup::new(d, k);
        let mut rng = thread_rng();
        let x = CyclotomicRingElement::<BenchFq>::random(&mut rng, d);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d={},k={}", d, k)),
            &(x, h),
            |b, (x, h)| b.iter(|| trace_naive(black_box(x), black_box(h))),
        );
    }

    group.finish();
}

fn bench_trace_tower(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_tower");

    for &(d, k) in TRACE_PARAMS {
        let h = GaloisSubgroup::new(d, k);
        let mut rng = thread_rng();
        let x = CyclotomicRingElement::<BenchFq>::random(&mut rng, d);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d={},k={}", d, k)),
            &(x, h),
            |b, (x, h)| b.iter(|| trace_tower(black_box(x), black_box(h))),
        );
    }

    group.finish();
}

fn bench_automorphism(c: &mut Criterion) {
    let mut group = c.benchmark_group("automorphism");

    for &d in AUTOMORPHISM_DIMS {
        for &sigma in AUTOMORPHISMS {
            let mut rng = thread_rng();
            let x = CyclotomicRingElement::<BenchFq>::random(&mut rng, d);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("d={},sigma={}", d, sigma),
                    format!("dim_{}_sigma_{}", d, sigma),
                ),
                &x,
                |b, x| b.iter(|| x.pow_automorphism(black_box(sigma))),
            );
        }
    }

    group.finish();
}

fn bench_trace_comparison_hachi_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_comparison");

    // Hachi parameters: d=1024, k=4
    let d = 1024;
    let k = 4;
    let h = GaloisSubgroup::new(d, k);
    let mut rng = thread_rng();
    let x = CyclotomicRingElement::<BenchFq>::random(&mut rng, d);

    group.bench_function("naive", |b| {
        b.iter(|| trace_naive(black_box(&x), black_box(&h)))
    });

    group.bench_function("tower", |b| {
        b.iter(|| trace_tower(black_box(&x), black_box(&h)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_trace_naive,
    bench_trace_tower,
    bench_automorphism,
    bench_trace_comparison_hachi_params
);
criterion_main!(benches);
