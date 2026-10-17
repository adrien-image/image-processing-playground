/**
 * The little linear algebra the machine-learning lessons need, on plain arrays.
 * Matrices are row-major Float64Arrays: A[i * cols + j].
 */

/** Solve A·x = b for a square n × n system by Gaussian elimination with partial pivoting. */
export function solve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const m = Float64Array.from(A), x = Float64Array.from(b);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r * n + col]) > Math.abs(m[pivot * n + col])) pivot = r;
    if (Math.abs(m[pivot * n + col]) < 1e-12) throw new Error('Singular matrix');
    if (pivot !== col) {
      for (let c = 0; c < n; c++) [m[col * n + c], m[pivot * n + c]] = [m[pivot * n + c], m[col * n + c]];
      [x[col], x[pivot]] = [x[pivot], x[col]];
    }
    for (let r = col + 1; r < n; r++) {
      const f = m[r * n + col] / m[col * n + col];
      if (f === 0) continue;
      for (let c = col; c < n; c++) m[r * n + c] -= f * m[col * n + c];
      x[r] -= f * x[col];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let c = r + 1; c < n; c++) s -= m[r * n + c] * x[c];
    x[r] = s / m[r * n + r];
  }
  return x;
}

/**
 * Eigenvalues and eigenvectors of a small symmetric n × n matrix (Jacobi rotations),
 * sorted by decreasing eigenvalue. vectors[k * n + i] is component i of eigenvector k.
 */
export function symmetricEigen(S: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(S);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq;
        a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk;
        a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = v[k * n + p], vkq = v[k * n + q];
        v[k * n + p] = c * vkp - s * vkq;
        v[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j * n + j] - a[i * n + i]);
  const values = Float64Array.from(order, i => a[i * n + i]);
  const vectors = new Float64Array(n * n);
  order.forEach((col, k) => { for (let i = 0; i < n; i++) vectors[k * n + i] = v[i * n + col]; });
  return { values, vectors };
}

/** The axes of the ellipse {x : (x − μ)ᵀ Σ⁻¹ (x − μ) = k²} of a 2 × 2 covariance [a, b; b, c]. */
export function covarianceEllipse(a: number, b: number, c: number, k = 2): { rx: number; ry: number; angle: number } {
  const mean = (a + c) / 2, diff = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const l1 = mean + diff, l2 = Math.max(mean - diff, 0);
  const angle = Math.abs(b) < 1e-12 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  return { rx: k * Math.sqrt(l1), ry: k * Math.sqrt(l2), angle };
}
