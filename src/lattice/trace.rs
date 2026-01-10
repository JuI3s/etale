//! Tower optimization for trace computation in cyclotomic rings
//!
//! For H = ⟨σ_{-1}, σ_{4k+1}⟩ ⊆ Aut(R_q), the trace Tr_H can be computed as:
//!   Tr_H(x) = (1 + σ_{-1})(1 + σ_{4k+1})(1 + σ_{(4k+1)²})...(x)
//!
//! This reduces O(|H|) automorphism applications to O(log|H|).
//!
//! For Hachi with d=1024, k=4: |H| = 256, so 256 → 8 operations (32× speedup)

use ark_ff::{Field, PrimeField};

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
        assert!(dim.is_power_of_two(), "dim must be power of two");
        let mut c = coeffs;
        c.resize(dim, F::zero());
        Self { coeffs: c, dim }
    }

    /// Create zero element
    pub fn zero(dim: usize) -> Self {
        Self::new(vec![F::zero(); dim], dim)
    }

    /// Create random element for testing
    pub fn random<R: rand::Rng>(rng: &mut R, dim: usize) -> Self
    where
        F: PrimeField,
    {
        let coeffs: Vec<F> = (0..dim).map(|_| F::rand(rng)).collect();
        Self::new(coeffs, dim)
    }

    /// Apply automorphism σ_i: X ↦ X^i
    ///
    /// For X^j, maps to X^{ij mod 2d} with sign from reduction mod (X^d + 1)
    pub fn pow_automorphism(&self, i: usize) -> Self {
        let d = self.dim;
        let two_d = 2 * d;
        let mut result = vec![F::zero(); d];

        for (j, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            // X^j ↦ X^{i*j mod 2d}
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
        assert_eq!(self.dim, other.dim);
        let coeffs: Vec<F> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Self::new(coeffs, self.dim)
    }
}

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
        assert!(d.is_power_of_two());
        assert!(k.is_power_of_two());
        assert!(d >= 2 * k, "k must divide d/2");
        Self { d, k, order: d / k }
    }

    /// Generate all elements of H = ⟨σ_{-1}, σ_{4k+1}⟩ ⊆ (Z/2d Z)^×
    ///
    /// By Gauss's theorem: (Z/2^α Z)^× ≅ Z/2Z × Z/2^{α-2}Z
    /// The map (a,b) ↦ (-1)^a · 3^b gives the isomorphism
    pub fn elements(&self) -> Vec<usize> {
        let two_d = 2 * self.d;
        let mut elements = Vec::with_capacity(self.order);

        // σ_{-1} generates {1, -1 mod 2d} = {1, 2d-1}
        // σ_{4k+1} has order d/(2k) in (Z/2d Z)^×

        let gen1 = two_d - 1; // -1 mod 2d
        let gen2 = 4 * self.k + 1;

        // Build H by enumeration
        let mut current = 1usize;
        let order_of_gen2 = self.d / (2 * self.k);

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

        elements.sort();
        elements
    }

    /// Get tower generators for computing trace via:
    /// Tr_H(x) = (1 + σ_{g_1})(1 + σ_{g_2})...(1 + σ_{g_t})(x)
    ///
    /// Returns generators in order for composition
    pub fn tower_generators(&self) -> Vec<usize> {
        let two_d = 2 * self.d;
        let mut generators = Vec::new();

        // First generator: σ_{-1}
        generators.push(two_d - 1);

        // Remaining generators: powers of σ_{4k+1}
        let base = 4 * self.k + 1;
        let num_steps = (self.order / 2).trailing_zeros() as usize;

        let mut power = base;
        for _ in 0..num_steps {
            generators.push(power);
            power = (power * power) % two_d;
        }

        generators
    }
}

/// Naive trace: Tr_H(x) = Σ_{σ∈H} σ(x)
///
/// Time: O(|H| * d) = O(d²/k)
pub fn trace_naive<F: Field>(
    x: &CyclotomicRingElement<F>,
    h: &GaloisSubgroup,
) -> CyclotomicRingElement<F> {
    let elements = h.elements();
    let mut result = CyclotomicRingElement::zero(x.dim);

    for sigma in elements {
        let sigma_x = x.pow_automorphism(sigma);
        result = result.add(&sigma_x);
    }

    result
}

/// Tower-optimized trace: Tr_H(x) = (1 + σ_{g_1})(1 + σ_{g_2})...(1 + σ_{g_t})(x)
///
/// Time: O(log|H| * d) = O(d * log(d/k))
///
/// This is the 32× speedup for Hachi parameters (d=1024, k=4)
pub fn trace_tower<F: Field>(
    x: &CyclotomicRingElement<F>,
    h: &GaloisSubgroup,
) -> CyclotomicRingElement<F> {
    let generators = h.tower_generators();
    let mut result = x.clone();

    // Apply (1 + σ_{g_i}) for each generator
    for gen in generators {
        let sigma_result = result.pow_automorphism(gen);
        result = result.add(&sigma_result);
    }

    result
}

/// Count operations for comparison
pub struct OpCount {
    pub automorphisms: usize,
    pub additions: usize,
}

pub fn count_naive_ops(h: &GaloisSubgroup) -> OpCount {
    OpCount {
        automorphisms: h.order,
        additions: h.order - 1,
    }
}

pub fn count_tower_ops(h: &GaloisSubgroup) -> OpCount {
    let num_generators = h.tower_generators().len();
    OpCount {
        automorphisms: num_generators,
        additions: num_generators,
    }
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

    println!("d={}, k={}, |H|={}", d, k, h.order);
    println!(
        "  Naive:  {} automorphisms, {:.1}µs",
        naive_ops.automorphisms, naive_time
    );
    println!(
        "  Tower:  {} automorphisms, {:.1}µs",
        tower_ops.automorphisms, tower_time
    );
    println!(
        "  Speedup: {:.1}× (theoretical: {}×)",
        naive_time / tower_time,
        naive_ops.automorphisms / tower_ops.automorphisms
    );
}

#[cfg(test)]
#[allow(non_local_definitions)]
mod test_config {
    use ark_ff::fields::models::fp::{Fp64, MontBackend, MontConfig};

    // Small prime field for testing: q ≡ 5 (mod 8)
    // Note: rust-analyzer may show a proc-macro panic error here, but this is a false positive.
    // The code compiles and runs correctly. Restart rust-analyzer if needed.
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
            vec![
                TestFq::from(1u64),
                TestFq::from(2u64),
                TestFq::from(3u64),
                TestFq::from(4u64),
            ],
            4,
        );
        let sigma_1 = x.pow_automorphism(1);
        assert_eq!(x, sigma_1);
    }

    #[test]
    fn test_automorphism_negation() {
        // σ_{-1}(X) = X^{-1} = -X^{d-1} in R_q
        let d = 4;
        let x = CyclotomicRingElement::<TestFq>::new(
            vec![
                TestFq::from(0u64),
                TestFq::from(1u64),
                TestFq::from(0u64),
                TestFq::from(0u64),
            ],
            d,
        );
        // X ↦ X^{-1 mod 8} = X^7 ≡ -X^3 mod (X^4 + 1)
        let sigma_neg1 = x.pow_automorphism(2 * d - 1);
        assert_eq!(sigma_neg1.coeffs[3], -TestFq::from(1u64));
    }

    #[test]
    fn test_trace_methods_agree() {
        let d = 64;
        let k = 4;
        let h = GaloisSubgroup::new(d, k);

        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let x = CyclotomicRingElement::<TestFq>::random(&mut rng, d);
            let naive = trace_naive(&x, &h);
            let tower = trace_tower(&x, &h);
            assert_eq!(naive, tower, "Naive and tower trace should match");
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
        let generators = h.tower_generators();
        // |H| = 256 = 2^8, so we need 8 generators
        assert_eq!(generators.len(), 8);
    }

    #[test]
    fn test_speedup_factor() {
        let h = GaloisSubgroup::new(1024, 4);
        let naive = count_naive_ops(&h);
        let tower = count_tower_ops(&h);

        // 256 / 8 = 32× theoretical speedup
        let speedup = naive.automorphisms / tower.automorphisms;
        assert_eq!(speedup, 32);
    }

    #[test]
    fn test_hachi_parameters() {
        // Hachi Figure 8: d=1024, k=4
        let d = 1024;
        let k = 4;
        let h = GaloisSubgroup::new(d, k);

        // |H| = d/k = 256
        assert_eq!(h.order, 256);

        // Tower has log_2(256) = 8 steps
        let gens = h.tower_generators();
        assert_eq!(gens.len(), 8);

        // First generator is σ_{-1}
        assert_eq!(gens[0], 2 * d - 1);

        // Second generator is σ_{4k+1} = σ_{17}
        assert_eq!(gens[1], 17);
    }
}
