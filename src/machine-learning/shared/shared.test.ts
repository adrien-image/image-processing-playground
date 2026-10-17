import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { accuracyOf, makeClassification, makeRegression, subset, trueCurve } from './datasets';
import { covarianceEllipse, solve, symmetricEigen } from './linalg';

describe('datasets', () => {
  it('makes balanced, repeatable 2-D problems inside the plot', () => {
    for (const shape of ['blobs', 'moons', 'circles', 'xor', 'spiral'] as const) {
      const a = makeClassification(shape, 200, 0.05, seededRandom(3), 3);
      const b = makeClassification(shape, 200, 0.05, seededRandom(3), 3);
      expect(a.xy).toEqual(b.xy);
      const counts = new Array(a.classes).fill(0);
      a.labels.forEach(l => counts[l]++);
      for (const c of counts) expect(c).toBeGreaterThan(200 / a.classes / 2);
      for (const v of a.xy) expect(Math.abs(v)).toBeLessThan(1.45);
    }
  });

  it('makes sorted regression data around the true curve', () => {
    const d = makeRegression(300, 0.1, seededRandom(1));
    for (let i = 1; i < d.n; i++) expect(d.x[i]).toBeGreaterThanOrEqual(d.x[i - 1]);
    const mse = d.x.reduce((s, x, i) => s + (d.y[i] - trueCurve(x)) ** 2, 0) / d.n;
    expect(mse).toBeGreaterThan(0.005);
    expect(mse).toBeLessThan(0.02);
  });

  it('subsets rows and scores predictions', () => {
    const d = makeClassification('blobs', 10, 0, seededRandom(1), 2);
    const s = subset(d, [3, 0]);
    expect(Array.from(s.xy)).toEqual([d.xy[6], d.xy[7], d.xy[0], d.xy[1]]);
    expect(accuracyOf([0, 1, 1], [0, 1, 0])).toBeCloseTo(2 / 3);
  });
});

describe('linalg', () => {
  it('solves a linear system that needs pivoting', () => {
    const A = Float64Array.from([0, 2, 1, 1, 1, 1, 2, 1, 0]);
    const x = solve(A, Float64Array.from([5, 4, 4]), 3);
    expect(Array.from(x).map(v => +v.toFixed(10))).toEqual([1, 2, 1]);
  });

  it('finds eigenpairs of a symmetric matrix, largest first', () => {
    const S = Float64Array.from([4, 1, 0, 1, 3, 1, 0, 1, 2]);
    const { values, vectors } = symmetricEigen(S, 3);
    expect(values[0]).toBeGreaterThan(values[1]);
    for (let k = 0; k < 3; k++) {
      const v = vectors.subarray(k * 3, k * 3 + 3);
      for (let i = 0; i < 3; i++) {
        const Sv = S[i * 3] * v[0] + S[i * 3 + 1] * v[1] + S[i * 3 + 2] * v[2];
        expect(Sv).toBeCloseTo(values[k] * v[i], 9);
      }
    }
    expect(values.reduce((a, b) => a + b, 0)).toBeCloseTo(9, 9);
  });

  it('turns a covariance into ellipse axes', () => {
    const e = covarianceEllipse(4, 0, 1, 1);
    expect(e.rx).toBeCloseTo(2);
    expect(e.ry).toBeCloseTo(1);
    expect(e.angle).toBeCloseTo(0);
    const d = covarianceEllipse(1, 0.9, 1, 1); // stretched along the diagonal
    expect(d.angle).toBeCloseTo(Math.PI / 4);
    expect(d.rx).toBeCloseTo(Math.sqrt(1.9));
  });
});
