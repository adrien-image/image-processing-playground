import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { makeClassification, type Shape2D } from '../shared/datasets';
import { contourSegments, joinSegments } from './contour';
import { decision, marginWidth, objective, primalWeights, supportVectors, trainSvm, type Kernel, type SvmModel } from './svm';

const signs = (labels: Uint8Array) => Array.from(labels, l => (l ? 1 : -1));
const f = (m: SvmModel, x: number, y: number) => decision(m, [x, y]);
const trainAccuracy = (m: SvmModel) => {
  let right = 0;
  for (let i = 0; i < m.y.length; i++) if (Math.sign(decision(m, m.x.subarray(2 * i, 2 * i + 2))) === m.y[i]) right++;
  return right / m.y.length;
};

/** The KKT conditions of the soft-margin dual, with SMO's tolerance: returns the points that break them. */
function kktViolations(m: SvmModel, tol = 2e-3): string[] {
  const bad: string[] = [];
  let balance = 0;
  for (let i = 0; i < m.y.length; i++) {
    const a = m.alpha[i], margin = m.y[i] * decision(m, m.x.subarray(2 * i, 2 * i + 2));
    balance += a * m.y[i];
    const ok = a < 0 || a > m.C ? false
      : a === 0 ? margin >= 1 - tol                          // outside the margin
        : a < m.C * (1 - 1e-8) ? Math.abs(margin - 1) < tol  // exactly on it
          : margin <= 1 + tol;                               // inside it or beyond
    if (!ok) bad.push(`point ${i}: α = ${a}, y·f = ${margin}`);
  }
  if (Math.abs(balance) > 1e-9 * m.C * m.y.length) bad.push(`Σαy = ${balance}`);
  return bad;
}
const expectKkt = (m: SvmModel) => expect(kktViolations(m)).toEqual([]);

describe('trainSvm, linear kernel', () => {
  it('puts the boundary of two points on their perpendicular bisector, with the margin equal to their distance', () => {
    const a = [0.3, -0.2], b = [-0.5, 0.4];
    const m = trainSvm([...a, ...b], [1, -1], 2, { type: 'linear' }, 100);
    expect(m.converged).toBe(true);
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    expect(f(m, mid[0], mid[1])).toBeCloseTo(0, 6);
    expect(f(m, a[0], a[1])).toBeCloseTo(1, 3);
    expect(f(m, b[0], b[1])).toBeCloseTo(-1, 3);
    // w is parallel to a − b, so the boundary is perpendicular to the segment.
    const w = primalWeights(m), d = [a[0] - b[0], a[1] - b[1]];
    expect(w[0] * d[1] - w[1] * d[0]).toBeCloseTo(0, 6);
    expect(marginWidth(m)).toBeCloseTo(Math.hypot(d[0], d[1]), 3);
    expect(supportVectors(m).support).toEqual([true, true]);
  });

  it('finds the widest gap between two rows of points, and ignores every point but the closest ones', () => {
    // Class +1 on the line y = 0.5 (and far above it), class −1 on y = −0.3: margin 0.8, boundary y = 0.1.
    const xs: number[] = [], ys: number[] = [];
    for (let k = -3; k <= 3; k++) { xs.push(k * 0.2, 0.5, k * 0.2, -0.3); ys.push(1, -1); }
    for (let k = 0; k < 5; k++) { xs.push(k * 0.1 - 0.2, 1 + k * 0.1); ys.push(1); }
    const m = trainSvm(xs, ys, 2, { type: 'linear' }, 1000);
    expect(marginWidth(m)).toBeCloseTo(0.8, 3);
    expect(f(m, 0.7, 0.1)).toBeCloseTo(0, 3);
    expect(f(m, -0.9, 0.1)).toBeCloseTo(0, 3);
    // None of the far points is a support vector...
    const { support } = supportVectors(m);
    for (let i = 14; i < 19; i++) expect(support[i]).toBe(false);
    // ...so training on the support vectors alone gives the same function.
    const keep = support.map((s, i) => (s ? i : -1)).filter(i => i >= 0);
    const m2 = trainSvm(keep.flatMap(i => [xs[2 * i], xs[2 * i + 1]]), keep.map(i => ys[i]), 2, { type: 'linear' }, 1000);
    for (const [x, y] of [[0, 0], [1, 1], [-1, 0.4]]) expect(f(m2, x, y)).toBeCloseTo(f(m, x, y), 3);
  });

  it('satisfies the KKT conditions on noisy blobs, and a smaller C gives a wider margin with more support vectors', () => {
    const data = makeClassification('blobs', 120, 0.1, seededRandom(4));
    const tight = trainSvm(data.xy, signs(data.labels), 2, { type: 'linear' }, 10);
    const loose = trainSvm(data.xy, signs(data.labels), 2, { type: 'linear' }, 0.05);
    expectKkt(tight);
    expectKkt(loose);
    expect(marginWidth(loose)).toBeGreaterThan(marginWidth(tight));
    const count = (m: SvmModel) => supportVectors(m).support.filter(Boolean).length;
    expect(count(loose)).toBeGreaterThan(count(tight));
    // Strong duality: the primal objective at w = Σ αᵢyᵢxᵢ equals the dual objective.
    for (const m of [tight, loose]) {
      const w = 2 / marginWidth(m), dual = m.alpha.reduce((s, a) => s + a, 0) - 0.5 * w * w;
      expect(objective(m).value).toBeCloseTo(dual, 1);
    }
  });
});

describe('trainSvm, kernels', () => {
  const kernels: Record<string, Kernel> = { rbf: { type: 'rbf', gamma: 5 }, poly: { type: 'poly', degree: 3 } };

  it.each(['circles', 'xor'] as Shape2D[])('separates %s with an RBF kernel, where a line cannot', shape => {
    const data = makeClassification(shape, 160, 0.05, seededRandom(9));
    const rbf = trainSvm(data.xy, signs(data.labels), 2, kernels.rbf, 10);
    const linear = trainSvm(data.xy, signs(data.labels), 2, { type: 'linear' }, 10);
    expect(rbf.converged).toBe(true);
    expect(trainAccuracy(rbf)).toBeGreaterThanOrEqual(0.98);
    expect(trainAccuracy(linear)).toBeLessThan(0.75);
    expectKkt(rbf);
  });

  it('satisfies the KKT conditions with a polynomial kernel on moons', () => {
    const data = makeClassification('moons', 120, 0.1, seededRandom(2));
    const m = trainSvm(data.xy, signs(data.labels), 2, kernels.poly, 1);
    expect(m.converged).toBe(true);
    expectKkt(m);
    expect(trainAccuracy(m)).toBeGreaterThan(0.9);
  });

  it('memorises with a very large γ: every point becomes a support vector', () => {
    const data = makeClassification('moons', 80, 0.25, seededRandom(5));
    const m = trainSvm(data.xy, signs(data.labels), 2, { type: 'rbf', gamma: 1000 }, 100);
    expect(trainAccuracy(m)).toBe(1);
    expect(supportVectors(m).support.every(Boolean)).toBe(true);
  });
});

describe('contourSegments', () => {
  it('traces the straight level set of a linear function', () => {
    const res = 20, extent = { x: [-1, 1] as [number, number], y: [-1, 1] as [number, number] };
    const values = new Float64Array(res * res);
    for (let row = 0; row < res; row++) for (let col = 0; col < res; col++) {
      const x = -1 + ((col + 0.5) / res) * 2;
      values[row * res + col] = x - 0.23;
    }
    const segs = contourSegments(values, res, 0, extent);
    expect(segs).toHaveLength(res - 1);
    for (const [x1, , x2] of segs) { expect(x1).toBeCloseTo(0.23, 9); expect(x2).toBeCloseTo(0.23, 9); }
  });
});

describe('joinSegments', () => {
  it('chains the segments of a closed contour into one polyline', () => {
    const res = 30, extent = { x: [-1, 1] as [number, number], y: [-1, 1] as [number, number] };
    const values = new Float64Array(res * res);
    for (let row = 0; row < res; row++) for (let col = 0; col < res; col++) {
      const x = -1 + ((col + 0.5) / res) * 2, y = 1 - ((row + 0.5) / res) * 2;
      values[row * res + col] = Math.hypot(x, y) - 0.6;
    }
    const segs = contourSegments(values, res, 0, extent);
    const lines = joinSegments(segs);
    expect(lines).toHaveLength(1);
    expect(lines[0].length / 2).toBe(segs.length + 1); // closed: first point repeated at the end
    for (let k = 0; k < lines[0].length; k += 2) expect(Math.hypot(lines[0][k], lines[0][k + 1])).toBeCloseTo(0.6, 1);
  });
});
