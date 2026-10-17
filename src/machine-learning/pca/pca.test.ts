import { describe, expect, it } from 'vitest';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { componentsFor, covariance, decode, encode, explainedVariance, fitPcaNow, orthonormalise, project, reconstruct, topEigen } from './pca';

/** A random orthonormal basis: rows of a d × d matrix. */
function randomBasis(d: number, random: () => number): Float64Array {
  const Q = Float64Array.from({ length: d * d }, () => gaussianRandom(random));
  orthonormalise(Q, d, d, random);
  return Q;
}

/** C = Qᵀ diag(λ) Q: eigenvectors are the rows of Q. */
function withSpectrum(Q: Float64Array, lambda: number[], d: number): Float64Array {
  const C = new Float64Array(d * d);
  for (let k = 0; k < d; k++) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i * d + j] += lambda[k] * Q[k * d + i] * Q[k * d + j];
  return C;
}

const run = <T>(gen: Generator<number, T>): T => {
  for (;;) {
    const s = gen.next();
    if (s.done) return s.value;
  }
};

/** Sparse-ish non-negative data (like pixel images), n × d. */
function sparseData(n: number, d: number, random: () => number): Float32Array {
  return Float32Array.from({ length: n * d }, (_, i) => (random() < 0.6 ? 0 : random() * (1 + (i % d) / d)));
}

describe('covariance', () => {
  it('matches the textbook formula (1/n) Σ (x − μ)(x − μ)ᵀ', () => {
    const n = 50, d = 7;
    const X = sparseData(n, d, seededRandom(1));
    const { mean, cov } = covariance(X, n, d);
    for (let i = 0; i < d; i++) {
      let m = 0;
      for (let r = 0; r < n; r++) m += X[r * d + i] / n;
      expect(mean[i]).toBeCloseTo(m, 12);
    }
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += (X[r * d + i] - mean[i]) * (X[r * d + j] - mean[j]);
      expect(cov[i * d + j]).toBeCloseTo(s / n, 12);
    }
  });
});

describe('topEigen (subspace iteration)', () => {
  it('recovers the eigenvalues and eigenvectors of a matrix with a known spectrum', () => {
    const d = 40, random = seededRandom(2);
    const Q = randomBasis(d, random);
    const lambda = Array.from({ length: d }, (_, i) => 10 * 0.8 ** i);
    const C = withSpectrum(Q, lambda, d);
    const { values, vectors } = run(topEigen(C, d, 8, { random, iterations: 30 }));
    for (let k = 0; k < 8; k++) {
      expect(values[k]).toBeCloseTo(lambda[k], 8);
      let overlap = 0;
      for (let i = 0; i < d; i++) overlap += vectors[k * d + i] * Q[k * d + i];
      expect(Math.abs(overlap)).toBeCloseTo(1, 8); // same direction, up to sign
    }
  });

  it('finds the directions of maximum variance of samples drawn with that covariance', () => {
    const d = 6, n = 20000, random = seededRandom(3);
    const Q = randomBasis(d, random);
    const sd = [3, 2, 1, 0.5, 0.3, 0.1];
    // x = Σₖ sdₖ zₖ qₖ with z standard normal has covariance Qᵀ diag(sd²) Q.
    const X = new Float64Array(n * d);
    for (let r = 0; r < n; r++) for (let k = 0; k < d; k++) {
      const z = sd[k] * gaussianRandom(random);
      for (let i = 0; i < d; i++) X[r * d + i] += z * Q[k * d + i];
    }
    const pca = fitPcaNow(X, n, d, 3, { random });
    for (let k = 0; k < 3; k++) {
      expect(pca.variances[k] / sd[k] ** 2).toBeGreaterThan(0.95);
      expect(pca.variances[k] / sd[k] ** 2).toBeLessThan(1.05);
      let overlap = 0;
      for (let i = 0; i < d; i++) overlap += pca.components[k * d + i] * Q[k * d + i];
      expect(Math.abs(overlap)).toBeGreaterThan(0.99);
    }
  });
});

describe('fitPca', () => {
  const n = 200, d = 30;
  const X = sparseData(n, d, seededRandom(4));

  it('returns orthonormal components with decreasing variances', () => {
    const pca = fitPcaNow(X, n, d, 12, { random: seededRandom(5) });
    for (let a = 0; a < 12; a++) for (let b = 0; b < 12; b++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += pca.components[a * d + i] * pca.components[b * d + i];
      expect(s).toBeCloseTo(a === b ? 1 : 0, 9);
    }
    for (let k = 1; k < 12; k++) expect(pca.variances[k]).toBeLessThanOrEqual(pca.variances[k - 1] + 1e-12);
  });

  it('rebuilds every sample exactly from all d components, and its variances sum to the total', () => {
    const pca = fitPcaNow(X, n, d, d, { random: seededRandom(6), oversample: 0, iterations: 60 });
    for (const r of [0, 17, 199]) {
      const x = X.subarray(r * d, (r + 1) * d);
      const back = reconstruct(pca, x, d);
      for (let i = 0; i < d; i++) expect(back[i]).toBeCloseTo(x[i], 9);
    }
    const { ratio, cumulative } = explainedVariance(pca);
    expect(ratio.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    expect(cumulative[d - 1]).toBeCloseTo(1, 9);
    expect(componentsFor(pca, 1)).toBeLessThanOrEqual(d);
  });

  it('projects on chosen axes consistently with encode, and decode inverts encode', () => {
    const pca = fitPcaNow(X, n, d, 5, { random: seededRandom(7) });
    const xy = project(pca, X, n, [0, 3]);
    const z = encode(pca, X.subarray(5 * d, 6 * d));
    expect(xy[10]).toBeCloseTo(z[0], 5);
    expect(xy[11]).toBeCloseTo(z[3], 5);
    // The coordinates of a decoded point are the ones we started from.
    const back = encode(pca, decode(pca, [1, -2, 0.5]), 3);
    expect([...back].map(v => Number(v.toFixed(9)))).toEqual([1, -2, 0.5]);
    // Fewer components, larger error (on average over the data).
    const err = (k: number) => {
      let s = 0;
      for (let r = 0; r < n; r++) {
        const x = X.subarray(r * d, (r + 1) * d), b = reconstruct(pca, x, k);
        for (let i = 0; i < d; i++) s += (x[i] - b[i]) ** 2;
      }
      return s;
    };
    expect(err(1)).toBeGreaterThan(err(3));
    expect(err(3)).toBeGreaterThan(err(5));
  });
});
