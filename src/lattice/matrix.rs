//! Matrix operations over Z_q

use super::vector::Vector;
use rand::Rng;

/// A matrix in Z_q^{m x n}
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: Vec<Vector>,
    pub modulus: i64,
}

impl Matrix {
    pub fn new(rows: Vec<Vector>) -> Self {
        let modulus = rows.first().map_or(1, |r| r.modulus);
        Matrix { rows, modulus }
    }

    pub fn random<R: Rng>(rng: &mut R, m: usize, n: usize, modulus: i64) -> Self {
        let rows: Vec<Vector> = (0..m)
            .map(|_| {
                let coeffs: Vec<i64> = (0..n).map(|_| rng.gen_range(0..modulus)).collect();
                Vector::new(coeffs, modulus)
            })
            .collect();
        Matrix::new(rows)
    }

    /// Matrix-vector product Av
    pub fn mul_vec(&self, v: &Vector) -> Vector {
        let coeffs: Vec<i64> = self.rows.iter().map(|row| row.inner_product(v)).collect();
        Vector::new(coeffs, self.modulus)
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn num_cols(&self) -> usize {
        self.rows.first().map_or(0, Vector::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_vector_mul() {
        let q = 101;
        let row1 = Vector::new(vec![1, 0, 0], q);
        let row2 = Vector::new(vec![0, 1, 0], q);
        let row3 = Vector::new(vec![0, 0, 1], q);
        let identity = Matrix::new(vec![row1, row2, row3]);

        let v = Vector::new(vec![5, 10, 15], q);
        let result = identity.mul_vec(&v);
        assert_eq!(result.coeffs, vec![5, 10, 15]);
    }
}
