/**
 * Principal component analysis (PCA): the directions along which the data varies most.
 *
 * For n samples x₁ … xₙ in d dimensions, with mean μ, the covariance matrix is
 *   C = (1/n) Σᵢ (xᵢ − μ)(xᵢ − μ)ᵀ          (d × d, symmetric, positive semi-definite)
 * Its eigenvectors v₁, v₂, … (sorted by eigenvalue λ₁ ≥ λ₂ ≥ …) are the principal
 * components: v₁ is the unit direction of maximum variance, v₂ the direction of maximum
 * variance orthogonal to v₁, and so on. λₖ is the variance of the data along vₖ, and the
 * eigenvalues add up to the total variance trace(C). Keeping the first k components,
 *   x ≈ μ + Σₖ ((x − μ) · vₖ) vₖ
 * is the best rank-k linear approximation of the data in squared error.
 *
 * For MNIST d = 784, and a full eigen-decomposition of the 784 × 784 covariance (Jacobi
 * rotations) takes far too long in a browser. We only need the top ~100 components, so we
 * use subspace iteration (block power iteration):
 *   1. start from p random vectors (a few more than we need: they speed up convergence),
 *   2. repeat: V ← orthonormalise(C · V)  — every power of C stretches the top directions
 *      more than the others (by λᵢ / λⱼ), Gram–Schmidt stops all columns collapsing onto v₁,
 *   3. Rayleigh–Ritz: diagonalise the small p × p matrix Vᵀ C V and rotate V accordingly,
 *      which sorts out the directions within the subspace that V has found.
 * Building C is cheap too, because MNIST is sparse (about 80 % of pixels are 0): we
 * accumulate Σ xxᵀ over the non-zero pixels of each image only, then subtract n·μμᵀ.
 *
 * Matrices are row-major. Data X is n × d; components are stored one per row (k × d).
 */
import { gaussianRandom } from '../../lib/random';
import { symmetricEigen } from '../shared/linalg';

export interface Pca {
  /** Dimension of the data. */
  d: number;
  /** Number of components kept. */
  k: number;
  mean: Float64Array;
  /** k × d: component c is components[c·d … c·d + d), a unit vector. */
  components: Float64Array;
  /** Variance along each component (the eigenvalues of C), decreasing. */
  variances: Float64Array;
  /** trace(C): the variance summed over all d directions. */
  totalVariance: number;
}

/** Mean and covariance C = (1/n) Σ (x − μ)(x − μ)ᵀ of the n × d rows of X (d × d, row-major). */
export function covariance(X: ArrayLike<number>, n: number, d: number): { mean: Float64Array; cov: Float64Array } {
  const mean = new Float64Array(d);
  const cov = new Float64Array(d * d);
  const nz = new Int32Array(d);
  const vals = new Float64Array(d);
  for (let r = 0; r < n; r++) {
    // Only the non-zero entries of a row contribute to xxᵀ.
    let m = 0;
    for (let i = 0; i < d; i++) {
      const v = X[r * d + i];
      mean[i] += v;
      if (v !== 0) { nz[m] = i; vals[m++] = v; }
    }
    for (let a = 0; a < m; a++) {
      const row = nz[a] * d, va = vals[a];
      for (let b = a; b < m; b++) cov[row + nz[b]] += va * vals[b];
    }
  }
  for (let i = 0; i < d; i++) mean[i] /= n;
  // Upper triangle → full symmetric matrix, minus the mean: C = Σxxᵀ/n − μμᵀ.
  for (let i = 0; i < d; i++) for (let j = i; j < d; j++) {
    const c = cov[i * d + j] / n - mean[i] * mean[j];
    cov[i * d + j] = c;
    cov[j * d + i] = c;
  }
  return { mean, cov };
}

/**
 * Orthonormalise the p rows of V (each of length d) in place by modified Gram–Schmidt,
 * run twice for accuracy ("twice is enough"). A row that is (numerically) a combination
 * of the previous ones is replaced by a random direction.
 */
export function orthonormalise(V: Float64Array, p: number, d: number, random: () => number): void {
  for (let pass = 0; pass < 2; pass++) {
    for (let a = 0; a < p; a++) {
      const va = V.subarray(a * d, (a + 1) * d);
      const before = Math.sqrt(dot(va, va));
      for (let b = 0; b < a; b++) {
        const vb = V.subarray(b * d, (b + 1) * d);
        const proj = dot(va, vb);
        for (let i = 0; i < d; i++) va[i] -= proj * vb[i];
      }
      let norm = Math.sqrt(dot(va, va));
      if (norm <= 1e-10 * (before || 1)) {
        for (let i = 0; i < d; i++) va[i] = gaussianRandom(random);
        a--; // orthogonalise the replacement too
        continue;
      }
      norm = 1 / norm;
      for (let i = 0; i < d; i++) va[i] *= norm;
    }
  }
}

/** W = V · C for p rows of V (= (C · Vᵀ)ᵀ, as C is symmetric). */
function timesCov(C: Float64Array, V: Float64Array, p: number, d: number): Float64Array {
  const W = new Float64Array(p * d);
  for (let a = 0; a < p; a++) {
    const v = V.subarray(a * d, (a + 1) * d), w = W.subarray(a * d, (a + 1) * d);
    for (let i = 0; i < d; i++) {
      const vi = v[i];
      if (vi === 0) continue;
      const row = C.subarray(i * d, (i + 1) * d);
      for (let j = 0; j < d; j++) w[j] += vi * row[j];
    }
  }
  return W;
}

export interface EigenOptions {
  /** Extra vectors iterated along with the k wanted ones (default 10). */
  oversample?: number;
  /** Power iterations (default 8). */
  iterations?: number;
  random: () => number;
}

/**
 * The top k eigenpairs of a symmetric positive semi-definite d × d matrix C by subspace
 * iteration with Rayleigh–Ritz. A generator: it yields its progress in [0, 1) after each
 * multiplication and each orthonormalisation, so a UI can spread the work over animation
 * frames, and returns the eigenvalues (decreasing) and eigenvectors (one per row).
 */
export function* topEigen(C: Float64Array, d: number, k: number, { oversample = 10, iterations = 8, random }: EigenOptions): Generator<number, { values: Float64Array; vectors: Float64Array }> {
  const p = Math.min(d, k + oversample);
  let V: Float64Array = new Float64Array(p * d);
  for (let i = 0; i < V.length; i++) V[i] = gaussianRandom(random);
  orthonormalise(V, p, d, random);
  for (let it = 0; it < iterations; it++) {
    V = timesCov(C, V, p, d);
    yield (it + 0.5) / (iterations + 1);
    orthonormalise(V, p, d, random);
    yield (it + 1) / (iterations + 1);
  }
  // Rayleigh–Ritz: B = V C Vᵀ is p × p; its eigenvectors rotate V onto the eigenvectors of C.
  const W = timesCov(C, V, p, d);
  const B = new Float64Array(p * p);
  for (let a = 0; a < p; a++) for (let b = a; b < p; b++) {
    const s = dot(V.subarray(a * d, (a + 1) * d), W.subarray(b * d, (b + 1) * d));
    B[a * p + b] = s;
    B[b * p + a] = s;
  }
  const small = symmetricEigen(B, p);
  const vectors = new Float64Array(k * d);
  for (let c = 0; c < k; c++) {
    const out = vectors.subarray(c * d, (c + 1) * d);
    for (let a = 0; a < p; a++) {
      const s = small.vectors[c * p + a];
      const v = V.subarray(a * d, (a + 1) * d);
      for (let i = 0; i < d; i++) out[i] += s * v[i];
    }
    // An eigenvector's sign is arbitrary: make its largest entry positive so pictures are stable.
    let big = 0;
    for (let i = 0; i < d; i++) if (Math.abs(out[i]) > Math.abs(big)) big = out[i];
    if (big < 0) for (let i = 0; i < d; i++) out[i] = -out[i];
  }
  return { values: Float64Array.from(small.values.subarray(0, k), v => Math.max(v, 0)), vectors };
}

/**
 * Fit PCA with k components to the n × d rows of X. A generator like topEigen (it yields
 * progress in [0, 1)); use fitPcaNow to run it in one go.
 */
export function* fitPca(X: ArrayLike<number>, n: number, d: number, k: number, options: EigenOptions): Generator<number, Pca> {
  const { mean, cov } = covariance(X, n, d);
  yield 0.1;
  let totalVariance = 0;
  for (let i = 0; i < d; i++) totalVariance += cov[i * d + i];
  const eig = topEigen(cov, d, k, options);
  for (let step = eig.next(); ; step = eig.next()) {
    if (step.done) return { d, k, mean, components: step.value.vectors, variances: step.value.values, totalVariance };
    yield 0.1 + 0.9 * step.value;
  }
}

/** Run fitPca to completion. */
export function fitPcaNow(X: ArrayLike<number>, n: number, d: number, k: number, options: EigenOptions): Pca {
  const it = fitPca(X, n, d, k, options);
  for (;;) {
    const step = it.next();
    if (step.done) return step.value;
  }
}

/** The coordinates (x − μ) · vₖ of one sample x on the first `count` components. */
export function encode(pca: Pca, x: ArrayLike<number>, count = pca.k): Float64Array {
  const { d, mean, components } = pca;
  const z = new Float64Array(count);
  for (let c = 0; c < count; c++) {
    let s = 0;
    for (let i = 0; i < d; i++) s += (x[i] - mean[i]) * components[c * d + i];
    z[c] = s;
  }
  return z;
}

/** μ + Σₖ zₖ vₖ over the first z.length components. */
export function decode(pca: Pca, z: ArrayLike<number>): Float64Array {
  const { d, components } = pca;
  const x = Float64Array.from(pca.mean);
  for (let c = 0; c < z.length; c++) {
    const zc = z[c];
    for (let i = 0; i < d; i++) x[i] += zc * components[c * d + i];
  }
  return x;
}

/** x rebuilt from its first k principal components. */
export function reconstruct(pca: Pca, x: ArrayLike<number>, k: number): Float64Array {
  return decode(pca, encode(pca, x, k));
}

/**
 * Project the n × d rows of X onto components `axes` (default the first two): an n × axes.length
 * Float32Array, e.g. (x, y) pairs for a scatter plot.
 */
export function project(pca: Pca, X: ArrayLike<number>, n: number, axes: number[] = [0, 1]): Float32Array {
  const { d, mean, components } = pca;
  const m = axes.length;
  const out = new Float32Array(n * m);
  for (let r = 0; r < n; r++) for (let a = 0; a < m; a++) {
    const base = axes[a] * d;
    let s = 0;
    for (let i = 0; i < d; i++) s += (X[r * d + i] - mean[i]) * components[base + i];
    out[r * m + a] = s;
  }
  return out;
}

/** Fraction of the total variance along each component, and the running (cumulative) sum. */
export function explainedVariance(pca: Pca): { ratio: number[]; cumulative: number[] } {
  const ratio = Array.from(pca.variances, v => v / pca.totalVariance);
  let s = 0;
  return { ratio, cumulative: ratio.map(r => (s += r)) };
}

/** The fewest components that explain at least `fraction` of the variance, or null if k is not enough. */
export function componentsFor(pca: Pca, fraction: number): number | null {
  const { cumulative } = explainedVariance(pca);
  const i = cumulative.findIndex(c => c >= fraction - 1e-12);
  return i < 0 ? null : i + 1;
}

/** Squared distance ‖a − b‖². */
export function squaredError(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
