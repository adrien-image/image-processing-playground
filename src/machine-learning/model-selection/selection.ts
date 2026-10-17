/**
 * Model selection on 1-D regression: polynomial fits of growing degree, train and
 * validation error, k-fold cross-validation, and the bias–variance decomposition.
 *
 * A degree-d polynomial is written in the Legendre basis P₀ … P_d on [−1, 1] rather than
 * as 1, x, x², …: the monomials look almost alike at high degree, so their least-squares
 * system is hopelessly ill-conditioned, while the Legendre polynomials are orthogonal on
 * [−1, 1] and the system stays well behaved up to degree 15 and beyond. The fit
 * minimises the mean squared error plus an optional ridge penalty,
 *   (1/n) Σᵢ (yᵢ − Σⱼ wⱼ Pⱼ(xᵢ))²  +  λ Σⱼ≥₁ wⱼ²
 * (the constant term is not penalised), through the normal equations
 *   (ΦᵀΦ / n + λ D) w = Φᵀy / n.
 *
 * Bias–variance: for a fixed x, average a method's prediction f̂(x) over many training
 * sets drawn from the same process. With y = f(x) + ε, Var ε = σ²,
 *   E[(f̂(x) − y)²] = (E f̂(x) − f(x))²  +  E[(f̂(x) − E f̂(x))²]  +  σ²
 *                    bias²                 variance                  noise
 * and the expected test error is the average of that over x.
 */
import { gaussianRandom } from '../../lib/random';
import { solve } from '../shared/linalg';
import { makeRegression, trueCurve, type Series1D } from '../shared/datasets';

export const MAX_DEGREE = 15;

/** A tiny ridge always added, so the system stays solvable with fewer points than coefficients. */
const JITTER = 1e-10;

/** P₀(x) … P_d(x) by Bonnet's recurrence (j + 1) P_{j+1} = (2j + 1) x P_j − j P_{j−1}. */
export function legendre(x: number, degree: number, out = new Float64Array(degree + 1)): Float64Array {
  out[0] = 1;
  if (degree >= 1) out[1] = x;
  for (let j = 1; j < degree; j++) out[j + 1] = ((2 * j + 1) * x * out[j] - j * out[j - 1]) / (j + 1);
  return out;
}

/** The value at x of the polynomial with Legendre coefficients w. */
export function evaluate(w: Float64Array, x: number): number {
  const p = legendre(x, w.length - 1);
  let s = 0;
  for (let j = 0; j < w.length; j++) s += w[j] * p[j];
  return s;
}

/** All indices 0 … n − 1. */
const everyRow = (n: number) => Array.from({ length: n }, (_, i) => i);

/** Least-squares (ridge) fit of a degree-d polynomial to the given rows of (x, y). */
export function fitPolynomial(data: Series1D, degree: number, lambda = 0, rows: readonly number[] = everyRow(data.n)): Float64Array {
  const d = degree + 1, m = rows.length;
  const A = new Float64Array(d * d), b = new Float64Array(d), p = new Float64Array(d);
  for (const i of rows) {
    legendre(data.x[i], degree, p);
    for (let r = 0; r < d; r++) {
      b[r] += (p[r] * data.y[i]) / m;
      for (let c = 0; c < d; c++) A[r * d + c] += (p[r] * p[c]) / m;
    }
  }
  for (let j = 0; j < d; j++) A[j * d + j] += JITTER + (j > 0 ? lambda : 0);
  return solve(A, b, d);
}

/** Mean squared error of the polynomial w on the given rows. */
export function mse(w: Float64Array, data: Series1D, rows: readonly number[] = everyRow(data.n)): number {
  let s = 0;
  for (const i of rows) s += (evaluate(w, data.x[i]) - data.y[i]) ** 2;
  return rows.length ? s / rows.length : 0;
}

/** k-fold split: shuffle the points, then deal them out to the folds in turn. fold[i] is point i's fold. */
export function kFolds(n: number, k: number, random: () => number): Uint8Array {
  const order = everyRow(n);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const fold = new Uint8Array(n);
  order.forEach((point, rank) => { fold[point] = rank % k; });
  return fold;
}

export interface CrossValidation {
  /** Mean of the k validation errors. */
  mean: number;
  /** Their standard error, std / √k. */
  se: number;
  perFold: number[];
  /** The k fits, each trained without its fold. */
  fits: Float64Array[];
}

/** k-fold cross-validation of a degree: k fits, each validated on the fold it did not see. */
export function crossValidate(data: Series1D, degree: number, fold: Uint8Array, k: number, lambda = 0): CrossValidation {
  const perFold: number[] = [], fits: Float64Array[] = [];
  for (let f = 0; f < k; f++) {
    const train: number[] = [], held: number[] = [];
    for (let i = 0; i < data.n; i++) (fold[i] === f ? held : train).push(i);
    const w = fitPolynomial(data, degree, lambda, train);
    fits.push(w);
    perFold.push(mse(w, data, held));
  }
  const mean = perFold.reduce((a, b) => a + b, 0) / k;
  const sd = Math.sqrt(perFold.reduce((a, e) => a + (e - mean) ** 2, 0) / Math.max(1, k - 1));
  return { mean, se: sd / Math.sqrt(k), perFold, fits };
}

export interface ErrorCurves {
  train: number[];
  validation: number[];
  cv: number[];
  cvSe: number[];
  /** The degree with the lowest cross-validation error. */
  cvBest: number;
}

/** Train, validation and k-fold CV error for every degree 0 … maxDegree. */
export function errorCurves(train: Series1D, validation: Series1D, fold: Uint8Array, k: number, lambda = 0, maxDegree = MAX_DEGREE): ErrorCurves {
  const out: ErrorCurves = { train: [], validation: [], cv: [], cvSe: [], cvBest: 0 };
  for (let d = 0; d <= maxDegree; d++) {
    const w = fitPolynomial(train, d, lambda);
    out.train.push(mse(w, train));
    out.validation.push(mse(w, validation));
    const cv = crossValidate(train, d, fold, k, lambda);
    out.cv.push(cv.mean);
    out.cvSe.push(cv.se);
  }
  out.cvBest = out.cv.indexOf(Math.min(...out.cv));
  return out;
}

export interface BiasVariance {
  /** One fit per fresh training set. */
  fits: Float64Array[];
  /** x positions (midpoints of equal strips of [−1, 1]) where the average fit, bias² and variance are measured. */
  grid: Float64Array;
  meanFit: Float64Array;
  bias2: number;
  variance: number;
  noise: number;
  /** Monte-Carlo expected squared error on fresh test points, averaged over the training sets (NaN without test points). */
  measured: number;
}

/**
 * Refit the same degree on `datasets` fresh training sets of n points from the same
 * process, and measure bias², variance and the expected test error directly.
 */
export function biasVariance(
  degree: number, lambda: number, n: number, noise: number, random: () => number,
  { datasets = 50, testPoints = 200, gridPoints = 200, f = trueCurve }: { datasets?: number; testPoints?: number; gridPoints?: number; f?: (x: number) => number } = {},
): BiasVariance {
  // Expectations over x ~ U[−1, 1] as averages over equal strips: a midpoint grid for bias² and
  // variance, one random test point per strip for the measured error (stratified sampling).
  const grid = Float64Array.from({ length: gridPoints }, (_, i) => -1 + (2 * (i + 0.5)) / gridPoints);
  const fits: Float64Array[] = [];
  const values: Float64Array[] = [];
  let measured = 0;
  for (let r = 0; r < datasets; r++) {
    const w = fitPolynomial(makeRegression(n, noise, random, f), degree, lambda);
    fits.push(w);
    values.push(Float64Array.from(grid, x => evaluate(w, x)));
    let e = 0;
    for (let t = 0; t < testPoints; t++) {
      const x = -1 + (2 * (t + random())) / testPoints, y = f(x) + noise * gaussianRandom(random);
      e += (evaluate(w, x) - y) ** 2;
    }
    measured += testPoints ? e / testPoints / datasets : NaN;
  }
  const meanFit = Float64Array.from(grid, (_, g) => values.reduce((a, v) => a + v[g], 0) / datasets);
  let bias2 = 0, variance = 0;
  grid.forEach((x, g) => {
    bias2 += (meanFit[g] - f(x)) ** 2;
    for (const v of values) variance += (v[g] - meanFit[g]) ** 2 / datasets;
  });
  return { fits, grid, meanFit, bias2: bias2 / gridPoints, variance: variance / gridPoints, noise: noise * noise, measured };
}
