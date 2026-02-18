//! Tower optimization for trace computation in cyclotomic rings
//!
//! For H = ⟨σ_{-1}, σ_{4k+1}⟩ ⊆ Aut(R_q), the trace Tr_H can be computed as:
//!   Tr_H(x) = (1 + σ_{-1})(1 + σ_{4k+1})(1 + σ_{(4k+1)²})...(x)
//!
//! This reduces O(|H|) automorphism applications to O(log|H|).
//!
//! For Hachi with d=1024, k=4: |H| = 256, so 256 → 8 operations (32× speedup)

use ark_ff::{Field, PrimeField};

// ============================================================================
// Cyclotomic Ring Element
// ============================================================================

/// Element in cyclotomic ring R_q = Z_q[X]/(X^d + 1)
#[derive(Clone, Debug, PartialEq)]
pub struct CyclotomicRingElement<F: Field> {
    /// Coefficients: c_0 + c_1*X + ... + c_{d-1}*X^{d-1}
    pub coeffs: Vec<F>,
    /// Ring dimension d = 2^α
    pub dim: usize,
}

impl<F: Field> CyclotomicRingElement<F> {
    /// Create new ring element, padding with zeros if needed
    pub fn new(coeffs: Vec<F>, dim: usize) -> Self {
        debug_assert!(dim.is_power_of_two(), "dim must be power of two");
        let mut c = coeffs;
        c.resize(dim, F::zero());
        Self { coeffs: c, dim }
    }

    /// Create zero element
    pub fn zero(dim: usize) -> Self {
        Self { coeffs: vec![F::zero(); dim], dim }
    }

    /// Create random element for testing
    pub fn random<R: rand::Rng>(rng: &mut R, dim: usize) -> Self
    where
        F: PrimeField,
    {
        let coeffs = (0..dim).map(|_| F::rand(rng)).collect();
        Self::new(coeffs, dim)
    }

    /// Apply automorphism σ_i: X ↦ X^i
    ///
    /// For X^j, maps to X^{ij mod 2d} with sign from reduction mod (X^d + 1)
    pub fn pow_automorphism(&self, i: usize) -> Self {
        let (d, two_d) = (self.dim, 2 * self.dim);
        let mut result = vec![F::zero(); d];

        for (j, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            let power = (i * j) % two_d;
            if power < d {
                result[power] += coeff;
            } else {
                // X^{d+k} ≡ -X^k mod (X^d + 1)
                result[power - d] -= coeff;
            }
        }

        Self::new(result, d)
    }

    /// Add two ring elements
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dim, other.dim);
        let coeffs = self.coeffs.iter()
            .zip(&other.coeffs)
            .map(|(a, b)| *a + *b)
            .collect();
        Self::new(coeffs, self.dim)
    }
}

// ============================================================================
// Galois Subgroup
// ============================================================================

/// Galois subgroup H = ⟨σ_{-1}, σ_{4k+1}⟩ for Hachi
pub struct GaloisSubgroup {
    /// Ring dimension d = 2^α
    pub d: usize,
    /// Extension field degree k (divides d/2)
    pub k: usize,
    /// |H| = d/k
    pub order: usize,
}

impl GaloisSubgroup {
    pub fn new(d: usize, k: usize) -> Self {
        debug_assert!(d.is_power_of_two());
        debug_assert!(k.is_power_of_two());
        debug_assert!(d >= 2 * k, "k must divide d/2");
        Self { d, k, order: d / k }
    }

    /// Generate all elements of H = ⟨σ_{-1}, σ_{4k+1}⟩ ⊆ (Z/2d Z)^×
    ///
    /// By Gauss's theorem: (Z/2^α Z)^× ≅ Z/2Z × Z/2^{α-2}Z
    pub fn elements(&self) -> Vec<usize> {
        let two_d = 2 * self.d;
        let (gen1, gen2) = (two_d - 1, 4 * self.k + 1); // σ_{-1}, σ_{4k+1}
        let order_of_gen2 = self.d / (2 * self.k);

        let mut elements = Vec::with_capacity(self.order);
        let mut current = 1usize;

        for _ in 0..2 {
            let mut power = current;
            for _ in 0..order_of_gen2 {
                if !elements.contains(&power) {
                    elements.push(power);
                }
                power = (power * gen2) % two_d;
            }
            current = (current * gen1) % two_d;
        }

        elements.sort_unstable();
        elements
    }

    /// Get tower generators for computing trace via:
    /// Tr_H(x) = (1 + σ_{g_1})(1 + σ_{g_2})...(1 + σ_{g_t})(x)
    pub fn tower_generators(&self) -> Vec<usize> {
        let two_d = 2 * self.d;
        let base = 4 * self.k + 1;
        let num_steps = (self.order / 2).trailing_zeros() as usize;

        // First generator: σ_{-1}, then powers of σ_{4k+1}
        let mut generators = vec![two_d - 1];
        let mut power = base;
        for _ in 0..num_steps {
            generators.push(power);
            power = (power * power) % two_d;
        }

        generators
    }
}

// ============================================================================
// Trace Computation
// ============================================================================

/// Naive trace: Tr_H(x) = Σ_{σ∈H} σ(x)
///
/// Time: O(|H| * d) = O(d²/k)
pub fn trace_naive<F: Field>(x: &CyclotomicRingElement<F>, h: &GaloisSubgroup) -> CyclotomicRingElement<F> {
    h.elements()
        .into_iter()
        .map(|sigma| x.pow_automorphism(sigma))
        .fold(CyclotomicRingElement::zero(x.dim), |acc, elem| acc.add(&elem))
}

/// Tower-optimized trace: Tr_H(x) = (1 + σ_{g_1})(1 + σ_{g_2})...(1 + σ_{g_t})(x)
///
/// Time: O(log|H| * d) = O(d * log(d/k))
///
/// This is the 32× speedup for Hachi parameters (d=1024, k=4)
pub fn trace_tower<F: Field>(x: &CyclotomicRingElement<F>, h: &GaloisSubgroup) -> CyclotomicRingElement<F> {
    h.tower_generators()
        .into_iter()
        .fold(x.clone(), |result, gen| {
            result.add(&result.pow_automorphism(gen))
        })
}

// ============================================================================
// Operation Counting
// ============================================================================

/// Count operations for comparison
pub struct OpCount {
    pub automorphisms: usize,
    pub additions: usize,
}

pub fn count_naive_ops(h: &GaloisSubgroup) -> OpCount {
    OpCount { automorphisms: h.order, additions: h.order - 1 }
}

pub fn count_tower_ops(h: &GaloisSubgroup) -> OpCount {
    let n = h.tower_generators().len();
    OpCount { automorphisms: n, additions: n }
}

/// Benchmark helper
pub fn benchmark_trace<F: Field + PrimeField>(d: usize, k: usize, trials: usize) {
    use std::time::Instant;

    let h = GaloisSubgroup::new(d, k);
    let mut rng = rand::thread_rng();

    // Warm up
    let x = CyclotomicRingElement::<F>::random(&mut rng, d);
    let _ = trace_naive(&x, &h);
    let _ = trace_tower(&x, &h);

    // Benchmark naive
    let start = Instant::now();
    for _ in 0..trials {
        let x = CyclotomicRingElement::<F>::random(&mut rng, d);
        let _ = trace_naive(&x, &h);
    }
    let naive_time = start.elapsed().as_micros() as f64 / trials as f64;

    // Benchmark tower
    let start = Instant::now();
    for _ in 0..trials {
        let x = CyclotomicRingElement::<F>::random(&mut rng, d);
        let _ = trace_tower(&x, &h);
    }
    let tower_time = start.elapsed().as_micros() as f64 / trials as f64;

    let naive_ops = count_naive_ops(&h);
    let tower_ops = count_tower_ops(&h);

    println!("d={d}, k={k}, |H|={}", h.order);
    println!("  Naive:  {} automorphisms, {naive_time:.1}µs", naive_ops.automorphisms);
    println!("  Tower:  {} automorphisms, {tower_time:.1}µs", tower_ops.automorphisms);
    println!(
        "  Speedup: {:.1}× (theoretical: {}×)",
        naive_time / tower_time,
        naive_ops.automorphisms / tower_ops.automorphisms
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_local_definitions)]
mod test_config {
    use ark_ff::fields::models::fp::{Fp64, MontBackend, MontConfig};

    #[derive(MontConfig)]
    #[modulus = "65557"]
    #[generator = "2"]
    #[small_subgroup_base = "0"]
    #[small_subgroup_power = "0"]
    pub struct TestFqConfig;

    pub type TestFq = Fp64<MontBackend<TestFqConfig, 1>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_config::TestFq;

    #[test]
    fn test_automorphism_identity() {
        let x = CyclotomicRingElement::<TestFq>::new(
            vec![TestFq::from(1u64), TestFq::from(2u64), TestFq::from(3u64), TestFq::from(4u64)],
            4,
        );
        assert_eq!(x, x.pow_automorphism(1));
    }

    #[test]
    fn test_automorphism_negation() {
        let d = 4;
        // X ↦ X^{-1 mod 8} = X^7 ≡ -X^3 mod (X^4 + 1)
        let x = CyclotomicRingElement::<TestFq>::new(
            vec![TestFq::from(0u64), TestFq::from(1u64), TestFq::from(0u64), TestFq::from(0u64)],
            d,
        );
        let sigma = x.pow_automorphism(2 * d - 1);
        assert_eq!(sigma.coeffs[3], -TestFq::from(1u64));
    }

    #[test]
    fn test_trace_methods_agree() {
        let (d, k) = (64, 4);
        let h = GaloisSubgroup::new(d, k);
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let x = CyclotomicRingElement::<TestFq>::random(&mut rng, d);
            assert_eq!(trace_naive(&x, &h), trace_tower(&x, &h));
        }
    }

    #[test]
    fn test_galois_subgroup_order() {
        let h = GaloisSubgroup::new(1024, 4);
        assert_eq!(h.order, 256);
        assert_eq!(h.elements().len(), 256);
    }

    #[test]
    fn test_tower_generators_count() {
        let h = GaloisSubgroup::new(1024, 4);
        assert_eq!(h.tower_generators().len(), 8); // |H| = 256 = 2^8
    }

    #[test]
    fn test_speedup_factor() {
        let h = GaloisSubgroup::new(1024, 4);
        let (naive, tower) = (count_naive_ops(&h), count_tower_ops(&h));
        assert_eq!(naive.automorphisms / tower.automorphisms, 32);
    }

    #[test]
    fn test_hachi_parameters() {
        let (d, k) = (1024, 4);
        let h = GaloisSubgroup::new(d, k);

        assert_eq!(h.order, 256);

        let gens = h.tower_generators();
        assert_eq!(gens.len(), 8);
        assert_eq!(gens[0], 2 * d - 1); // σ_{-1}
        assert_eq!(gens[1], 17);        // σ_{17}
    }
}
