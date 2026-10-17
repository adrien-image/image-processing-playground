import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import {
  cluster, fitLogistic, leastSquares, lineCurvature, lineDescent, lineGradient, lineMse, logisticGradient, logLoss,
  normalAngle, noisyLine, predictLinear, predictProba, sigmoid, thresholdAccuracy, twoBlobs,
} from './linear';
import { clipLine, contourSegments } from './surface';

const last = <T,>(a: T[]) => a[a.length - 1];

describe('least squares', () => {
  it('recovers a noiseless plane exactly', () => {
    const random = seededRandom(1);
    const X = Array.from({ length: 40 }, () => random() * 2 - 1);
    const y = Array.from({ length: 20 }, (_, i) => 1.5 * X[2 * i] - 0.7 * X[2 * i + 1] + 0.3);
    const m = leastSquares(X, y, 2);
    expect(m.w[0]).toBeCloseTo(1.5, 10);
    expect(m.w[1]).toBeCloseTo(-0.7, 10);
    expect(m.b).toBeCloseTo(0.3, 10);
  });

  it('matches the 1-D textbook formula w = cov(x, y) / var(x), b = ȳ − w·x̄', () => {
    const { x, y } = noisyLine(30, seededRandom(4));
    const mx = x.reduce((a, b) => a + b) / x.length, my = y.reduce((a, b) => a + b) / y.length;
    let cov = 0, vr = 0;
    x.forEach((v, i) => { cov += (v - mx) * (y[i] - my); vr += (v - mx) ** 2; });
    const m = leastSquares(x, y, 1);
    expect(m.w[0]).toBeCloseTo(cov / vr, 10);
    expect(m.b).toBeCloseTo(my - (cov / vr) * mx, 10);
  });

  it('is the minimum: the gradient vanishes there and nudging (w, b) only raises the MSE', () => {
    const { x, y } = noisyLine(30, seededRandom(4));
    const m = leastSquares(x, y, 1);
    const [gw, gb] = lineGradient(x, y, m.w[0], m.b);
    expect(Math.hypot(gw, gb)).toBeLessThan(1e-10);
    const best = lineMse(x, y, m.w[0], m.b);
    for (const [dw, db] of [[0.01, 0], [0, 0.01], [-0.01, 0.01]]) expect(lineMse(x, y, m.w[0] + dw, m.b + db)).toBeGreaterThan(best);
  });

  it('survives collinear inputs (all x equal) instead of throwing', () => {
    const m = leastSquares([1, 1, 1], [0, 1, 2], 1);
    expect(Number.isFinite(m.w[0]) && Number.isFinite(m.b)).toBe(true);
    expect(m.w[0] + m.b).toBeCloseTo(1, 6);
  });

  it('one outlier at (2.4, −0.9) halves the slope of the lesson\'s line (0.62 → 0.29)', () => {
    const { x, y } = noisyLine(30, seededRandom(4));
    const before = leastSquares(x, y, 1).w[0];
    const after = leastSquares([...x, 2.4], [...y, -0.9], 1).w[0];
    expect(before).toBeCloseTo(0.62, 2);
    expect(after).toBeCloseTo(0.29, 2);
  });
});

describe('gradient descent on (w, b)', () => {
  const { x, y } = noisyLine(30, seededRandom(4));
  const exact = leastSquares(x, y, 1);
  const [lmax, lmin] = lineCurvature(x);

  it('the gradient matches finite differences', () => {
    const [gw, gb] = lineGradient(x, y, -0.3, 1.2), h = 1e-6;
    expect(gw).toBeCloseTo((lineMse(x, y, -0.3 + h, 1.2) - lineMse(x, y, -0.3 - h, 1.2)) / (2 * h), 6);
    expect(gb).toBeCloseTo((lineMse(x, y, -0.3, 1.2 + h) - lineMse(x, y, -0.3, 1.2 - h)) / (2 * h), 6);
  });

  it('the bowl is narrow for x in [0, 2]: curvatures 4.68 and 0.26, so η must stay below 0.43', () => {
    expect(lmax).toBeCloseTo(4.68, 2);
    expect(lmin).toBeCloseTo(0.26, 2);
    expect(2 / lmax).toBeCloseTo(0.43, 2);
  });

  it('converges to the closed-form solution below 2/λmax', () => {
    for (const lr of [0.1, 0.3, 0.4]) {
      const run = lineDescent(x, y, [-1, 2], lr, 3000);
      expect(run.diverged).toBe(false);
      const [w, b] = last(run.path);
      expect(w).toBeCloseTo(exact.w[0], 6);
      expect(b).toBeCloseTo(exact.b, 6);
    }
  });

  it('takes 97 steps to get within 10⁻³ at η = 0.3, and diverges just above 2/λmax', () => {
    const run = lineDescent(x, y, [-1, 2], 0.3, 300);
    expect(run.path.findIndex(([w, b]) => Math.hypot(w - exact.w[0], b - exact.b) < 1e-3)).toBe(97);
    expect(lineDescent(x, y, [-1, 2], (2 / lmax) * 1.05, 2000).diverged).toBe(true);
    expect(lineDescent(x, y, [-1, 2], (2 / lmax) * 0.95, 2000).diverged).toBe(false);
  });
});

describe('logistic regression', () => {
  it('sigmoid is stable at large |z| and logLoss(½) = ln 2', () => {
    expect(sigmoid(800)).toBe(1);
    expect(sigmoid(-800)).toBe(0);
    expect(sigmoid(0)).toBe(0.5);
    expect(logLoss([0.5, 0.5], [0, 1])).toBeCloseTo(Math.LN2, 12);
  });

  it('the gradient (1/n)Aᵀ(p − y) matches finite differences of the log-loss', () => {
    const { xy, labels } = twoBlobs(40, 0.8, 0.4, seededRandom(2));
    const model = { w: Float64Array.from([0.7, -0.4]), b: 0.2 };
    const g = logisticGradient(model, xy, labels, 2), h = 1e-6;
    const loss = (w0: number, w1: number, b: number) => logLoss(predictProba({ w: Float64Array.from([w0, w1]), b }, xy, 2), labels);
    expect(g[0]).toBeCloseTo((loss(0.7 + h, -0.4, 0.2) - loss(0.7 - h, -0.4, 0.2)) / (2 * h), 6);
    expect(g[1]).toBeCloseTo((loss(0.7, -0.4 + h, 0.2) - loss(0.7, -0.4 - h, 0.2)) / (2 * h), 6);
    expect(g[2]).toBeCloseTo((loss(0.7, -0.4, 0.2 + h) - loss(0.7, -0.4, 0.2 - h)) / (2 * h), 6);
  });

  it('gradient descent lowers the loss every step and classifies the blobs', () => {
    const { xy, labels } = twoBlobs(200, 1.2, 0.3, seededRandom(3));
    const fit = fitLogistic(xy, labels, 2, { lr: 1, steps: 300 });
    for (let i = 1; i < fit.losses.length; i++) expect(fit.losses[i]).toBeLessThanOrEqual(fit.losses[i - 1] + 1e-12);
    expect(fit.losses[0]).toBeCloseTo(Math.LN2, 12);
    expect(thresholdAccuracy(predictProba(fit, xy, 2), labels, 0.5)).toBeGreaterThan(0.95);
  });

  it('far points of the right class rotate the least-squares boundary, not the logistic one', () => {
    const { xy, labels } = twoBlobs(100, 1.2, 0.25, seededRandom(9));
    const farXy = [...xy, ...cluster(1.4, 2.8, 10, seededRandom(5))], farLabels = [...labels, ...Array<number>(10).fill(1)];
    const opts = { lr: 1, steps: 1000, l2: 1e-3 };
    const logit = fitLogistic(xy, labels, 2, opts), logitFar = fitLogistic(farXy, farLabels, 2, opts);
    const ls = leastSquares(xy, labels, 2), lsFar = leastSquares(farXy, farLabels, 2);
    expect(Math.abs(normalAngle(logitFar) - normalAngle(logit))).toBeLessThan(1);
    expect(Math.abs(normalAngle(lsFar) - normalAngle(ls))).toBeGreaterThan(20);
    // Judged on the original points: least squares now misclassifies some, logistic none.
    expect(thresholdAccuracy(predictProba(logitFar, xy, 2), labels, 0.5)).toBe(1);
    expect(thresholdAccuracy(predictLinear(ls, xy, 2), labels, 0.5)).toBe(1);
    expect(thresholdAccuracy(predictLinear(lsFar, xy, 2), labels, 0.5)).toBeLessThan(0.97);
  });
});

describe('plot helpers', () => {
  it('clipLine keeps the part of a line inside the box', () => {
    const box = { x: [-1, 1] as [number, number], y: [-1, 1] as [number, number] };
    expect(clipLine(1, -1, 0, box)).toEqual([-1, -1, 1, 1]); // x = y, corner to corner
    expect(clipLine(0, 1, 0.5, box)).toEqual([-1, 0.5, 1, 0.5]); // y = 0.5
    expect(clipLine(1, 1, 5, box)).toBeNull();
  });

  it('contourSegments traces a circle at the right radius', () => {
    const res = 41, values = new Float64Array(res * res);
    for (let r = 0; r < res; r++) for (let c = 0; c < res; c++) values[r * res + c] = Math.hypot(r - 20, c - 20);
    const segs = contourSegments(values, res, 10);
    expect(segs.length).toBeGreaterThan(40 * 4);
    for (let k = 0; k < segs.length; k += 2) expect(Math.hypot(segs[k] - 20, segs[k + 1] - 20)).toBeCloseTo(10, 0);
  });
});
