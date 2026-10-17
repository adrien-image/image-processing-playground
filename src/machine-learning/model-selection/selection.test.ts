import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { makeRegression } from '../shared/datasets';
import { biasVariance, crossValidate, errorCurves, evaluate, fitPolynomial, kFolds, legendre, MAX_DEGREE, mse } from './selection';

describe('polynomial least squares in the Legendre basis', () => {
  it('matches the closed forms and is orthogonal on [−1, 1]', () => {
    const p = legendre(0.3, 3);
    expect(p[2]).toBeCloseTo((3 * 0.09 - 1) / 2, 12);
    expect(p[3]).toBeCloseTo((5 * 0.027 - 3 * 0.3) / 2, 12);
    // ∫ P₄ P₇ dx = 0 and ∫ P₅² dx = 2 / 11, by the midpoint rule.
    let cross = 0, square = 0;
    for (let i = 0; i < 20000; i++) {
      const q = legendre(-1 + (i + 0.5) / 10000, 7);
      cross += (q[4] * q[7]) / 10000; square += (q[5] * q[5]) / 10000;
    }
    expect(cross).toBeCloseTo(0, 6);
    expect(square).toBeCloseTo(2 / 11, 6);
  });

  it('recovers a polynomial exactly from noiseless data', () => {
    const f = (x: number) => 0.5 - x + 2 * x ** 3;
    const data = makeRegression(25, 0, seededRandom(1), f);
    const w = fitPolynomial(data, 5);
    for (const x of [-0.9, 0, 0.42]) expect(evaluate(w, x)).toBeCloseTo(f(x), 6);
  });

  it('never has a higher training error at a higher degree', () => {
    for (const n of [12, 30, 100]) {
      const data = makeRegression(n, 0.3, seededRandom(n));
      const errors = Array.from({ length: MAX_DEGREE + 1 }, (_, d) => mse(fitPolynomial(data, d), data));
      for (let d = 1; d <= MAX_DEGREE; d++) expect(errors[d]).toBeLessThanOrEqual(errors[d - 1] * (1 + 1e-6) + 1e-12);
      expect(errors[MAX_DEGREE]).toBeLessThan(errors[0] / 3);
    }
  });
});

describe('k-fold cross-validation', () => {
  it('splits the points into k disjoint folds of nearly equal size that cover everything', () => {
    for (const [n, k] of [[30, 5], [31, 10], [200, 7], [10, 2]]) {
      const fold = kFolds(n, k, seededRandom(n + k));
      expect(fold).toHaveLength(n);
      const sizes = new Array(k).fill(0);
      fold.forEach(f => { expect(f).toBeLessThan(k); sizes[f]++; });
      expect(Math.max(...sizes) - Math.min(...sizes)).toBeLessThanOrEqual(1);
      expect(sizes.reduce((a, b) => a + b, 0)).toBe(n);
    }
  });

  it('validates each fit only on the fold it was not trained on', () => {
    const data = makeRegression(40, 0.2, seededRandom(3));
    const fold = kFolds(40, 4, seededRandom(4));
    const cv = crossValidate(data, 3, fold, 4);
    for (let f = 0; f < 4; f++) {
      const held = Array.from({ length: 40 }, (_, i) => i).filter(i => fold[i] === f);
      const train = Array.from({ length: 40 }, (_, i) => i).filter(i => fold[i] !== f);
      expect(cv.perFold[f]).toBeCloseTo(mse(fitPolynomial(data, 3, 0, train), data, held), 12);
    }
  });

  it('picks a moderate degree for the sine data, never 0 or 15', () => {
    const chosen: number[] = [];
    for (let seed = 1; seed <= 15; seed++) {
      const data = makeRegression(60, 0.25, seededRandom(seed));
      const curves = errorCurves(data, makeRegression(60, 0.25, seededRandom(seed + 100)), kFolds(60, 5, seededRandom(seed + 200)), 5);
      chosen.push(curves.cvBest);
      // The U: validation error falls from degree 0 to the chosen degree, then rises again by degree 15.
      expect(curves.validation[curves.cvBest]).toBeLessThan(curves.validation[0]);
      expect(curves.validation[MAX_DEGREE]).toBeGreaterThan(Math.min(...curves.validation));
    }
    expect(chosen.every(d => d > 0 && d < MAX_DEGREE)).toBe(true);
    const sorted = [...chosen].sort((a, b) => a - b);
    expect(sorted[7]).toBeGreaterThanOrEqual(3);
    expect(sorted[7]).toBeLessThanOrEqual(9);
    expect(chosen.filter(d => d >= 3 && d <= 9).length).toBeGreaterThanOrEqual(12);
  });
});

describe('bias–variance decomposition', () => {
  it('bias² + variance + noise matches the Monte-Carlo expected test error', () => {
    for (const degree of [1, 3, 6]) {
      const bv = biasVariance(degree, 0, 40, 0.3, seededRandom(degree), { datasets: 100, testPoints: 400 });
      const sum = bv.bias2 + bv.variance + bv.noise;
      expect(Math.abs(sum - bv.measured) / bv.measured).toBeLessThan(0.05);
    }
  });

  it('trades bias for variance with the degree, and back with the ridge λ', () => {
    const at = (d: number, lambda = 0) => biasVariance(d, lambda, 30, 0.3, seededRandom(5), { testPoints: 0 });
    const [d1, d3, d10] = [at(1), at(3), at(10)];
    expect(d1.bias2).toBeGreaterThan(10 * d3.bias2);
    expect(d10.variance).toBeGreaterThan(3 * d3.variance);
    // A little ridge tames degree 10; a lot of it flattens the fit: less variance, more bias.
    const [light, heavy] = [at(10, 1e-3), at(10, 0.3)];
    expect(light.variance).toBeLessThan(d10.variance / 3);
    expect(heavy.variance).toBeLessThan(light.variance);
    expect(heavy.bias2).toBeGreaterThan(3 * light.bias2);
  });
});
