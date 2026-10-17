/**
 * Linear and logistic regression, from scratch.
 *
 * A linear model scores an input x (d features) as  f(x) = w·x + b.
 *
 * Regression fits f to real targets y by least squares, minimising the mean squared error
 *   MSE(w, b) = (1/n) Σ (w·xᵢ + b − yᵢ)².
 * The MSE is a quadratic bowl in (w, b), so its minimum has a closed form: setting the
 * gradient to zero gives the normal equations  AᵀA θ = Aᵀy,  where A is X with a column
 * of ones appended and θ = (w, b). Gradient descent reaches the same point step by step:
 *   ∇MSE = (2/n) Aᵀ(Aθ − y),   θ ← θ − η·∇MSE.
 * Its Hessian H = (2/n) AᵀA is constant, so gradient descent converges exactly when
 * η < 2 / λmax(H); a long, narrow bowl (λmax ≫ λmin) makes it zig-zag across the valley
 * while it crawls along it.
 *
 * Logistic regression turns the score into a probability for binary labels y ∈ {0, 1}:
 *   p(x) = σ(w·x + b),  σ(z) = 1 / (1 + e⁻ᶻ),
 * and minimises the cross-entropy (log-loss)
 *   L = −(1/n) Σ [yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)],   ∇L = (1/n) Aᵀ(p − y).
 * There is no closed form, but L is convex, so gradient descent finds the minimum.
 *
 * Inputs are flat row-major arrays: X[i·d + j] is feature j of example i.
 */
import { gaussianRandom } from '../../lib/random';
import { solve } from '../shared/linalg';

export interface LinearModel {
  w: Float64Array;
  b: number;
}

/** f(xᵢ) = w·xᵢ + b for every row of X. */
export function predictLinear(model: LinearModel, X: ArrayLike<number>, d: number): Float64Array {
  const n = X.length / d;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = model.b;
    for (let j = 0; j < d; j++) s += model.w[j] * X[i * d + j];
    out[i] = s;
  }
  return out;
}

/**
 * The exact least-squares fit, from the normal equations AᵀA θ = Aᵀy (a (d+1) × (d+1)
 * system). A tiny ridge keeps it solvable if the columns are collinear (e.g. all x equal).
 */
export function leastSquares(X: ArrayLike<number>, y: ArrayLike<number>, d: number): LinearModel {
  const n = y.length, k = d + 1;
  const AtA = new Float64Array(k * k), Aty = new Float64Array(k);
  const row = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) row[j] = X[i * d + j];
    row[d] = 1;
    for (let r = 0; r < k; r++) {
      Aty[r] += row[r] * y[i];
      for (let c = 0; c < k; c++) AtA[r * k + c] += row[r] * row[c];
    }
  }
  let theta: Float64Array;
  try {
    theta = solve(AtA, Aty, k);
  } catch {
    for (let r = 0; r < k; r++) AtA[r * k + r] += 1e-9 * n;
    theta = solve(AtA, Aty, k);
  }
  return { w: theta.slice(0, d), b: theta[d] };
}

/** Mean squared error of the model on (X, y). */
export function mse(model: LinearModel, X: ArrayLike<number>, y: ArrayLike<number>, d: number): number {
  const f = predictLinear(model, X, d);
  let s = 0;
  for (let i = 0; i < y.length; i++) s += (f[i] - y[i]) ** 2;
  return y.length ? s / y.length : 0;
}

// ---- 1-D regression: the (w, b) plane the lesson draws ----

/** MSE of the line y = w·x + b. */
export function lineMse(x: ArrayLike<number>, y: ArrayLike<number>, w: number, b: number): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (w * x[i] + b - y[i]) ** 2;
  return x.length ? s / x.length : 0;
}

/** ∇MSE = (2/n) Σ rᵢ·(xᵢ, 1), with residuals rᵢ = w·xᵢ + b − yᵢ. */
export function lineGradient(x: ArrayLike<number>, y: ArrayLike<number>, w: number, b: number): [number, number] {
  let gw = 0, gb = 0;
  for (let i = 0; i < x.length; i++) {
    const r = w * x[i] + b - y[i];
    gw += r * x[i];
    gb += r;
  }
  const n = Math.max(1, x.length);
  return [(2 * gw) / n, (2 * gb) / n];
}

/** One gradient-descent step on the line's (w, b). */
export function lineStep(x: ArrayLike<number>, y: ArrayLike<number>, w: number, b: number, lr: number): [number, number] {
  const [gw, gb] = lineGradient(x, y, w, b);
  return [w - lr * gw, b - lr * gb];
}

/**
 * The eigenvalues (largest first) of the MSE's Hessian H = (2/n)[[Σx², Σx], [Σx, n]].
 * Gradient descent on (w, b) converges if and only if η < 2 / λmax.
 */
export function lineCurvature(x: ArrayLike<number>): [number, number] {
  const n = Math.max(1, x.length);
  let sx = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) { sx += x[i]; sxx += x[i] * x[i]; }
  const a = (2 * sxx) / n, c = 2, off = (2 * sx) / n;
  const mean = (a + c) / 2, diff = Math.sqrt(((a - c) / 2) ** 2 + off * off);
  return [mean + diff, mean - diff];
}

/**
 * `steps` steps of gradient descent from `start`, stopping early if the iterate blows up.
 * Returns the path of (w, b), including the start.
 */
export function lineDescent(x: ArrayLike<number>, y: ArrayLike<number>, start: [number, number], lr: number, steps: number): { path: [number, number][]; diverged: boolean } {
  const path: [number, number][] = [start];
  let [w, b] = start;
  for (let s = 0; s < steps; s++) {
    [w, b] = lineStep(x, y, w, b, lr);
    if (!Number.isFinite(w) || !Number.isFinite(b) || Math.abs(w) + Math.abs(b) > 1e6) return { path, diverged: true };
    path.push([w, b]);
  }
  return { path, diverged: false };
}

// ---- Logistic regression ----

/** σ(z) = 1 / (1 + e⁻ᶻ), written so that neither branch overflows. */
export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export interface LogisticFit extends LinearModel {
  /** The training log-loss before each step, then after the last one. */
  losses: number[];
}

export interface LogisticOptions {
  /** Learning rate η of full-batch gradient descent. */
  lr?: number;
  steps?: number;
  /** L2 penalty λ·|w|²/2 (not on b); a little keeps w finite on separable data. */
  l2?: number;
  /** Start from these parameters instead of zeros (warm start). */
  init?: LinearModel;
}

/** P(y = 1 | x) for every row of X. */
export function predictProba(model: LinearModel, X: ArrayLike<number>, d: number): Float64Array {
  const f = predictLinear(model, X, d);
  for (let i = 0; i < f.length; i++) f[i] = sigmoid(f[i]);
  return f;
}

/** The mean cross-entropy of probabilities p against labels y ∈ {0, 1} (clipped away from log 0). */
export function logLoss(p: ArrayLike<number>, y: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) {
    const q = Math.min(1 - 1e-12, Math.max(1e-12, p[i]));
    s -= y[i] ? Math.log(q) : Math.log(1 - q);
  }
  return y.length ? s / y.length : 0;
}

/** The log-loss gradient (1/n) Aᵀ(p − y) (+ λw), as [∂w₀, …, ∂w_{d−1}, ∂b]. */
export function logisticGradient(model: LinearModel, X: ArrayLike<number>, y: ArrayLike<number>, d: number, l2 = 0): Float64Array {
  const n = y.length;
  const p = predictProba(model, X, d);
  const g = new Float64Array(d + 1);
  for (let i = 0; i < n; i++) {
    const r = p[i] - y[i];
    for (let j = 0; j < d; j++) g[j] += r * X[i * d + j];
    g[d] += r;
  }
  for (let j = 0; j <= d; j++) g[j] /= Math.max(1, n);
  for (let j = 0; j < d; j++) g[j] += l2 * model.w[j];
  return g;
}

/** Fit logistic regression by full-batch gradient descent on the log-loss. */
export function fitLogistic(X: ArrayLike<number>, y: ArrayLike<number>, d: number, { lr = 1, steps = 500, l2 = 0, init }: LogisticOptions = {}): LogisticFit {
  const model: LinearModel = init ? { w: Float64Array.from(init.w), b: init.b } : { w: new Float64Array(d), b: 0 };
  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    losses.push(logLoss(predictProba(model, X, d), y));
    const g = logisticGradient(model, X, y, d, l2);
    for (let j = 0; j < d; j++) model.w[j] -= lr * g[j];
    model.b -= lr * g[d];
  }
  losses.push(logLoss(predictProba(model, X, d), y));
  return { ...model, losses };
}

/** Fraction of rows where (score ≥ threshold) matches the 0/1 label. */
export function thresholdAccuracy(scores: ArrayLike<number>, y: ArrayLike<number>, threshold: number): number {
  let right = 0;
  for (let i = 0; i < y.length; i++) if ((scores[i] >= threshold ? 1 : 0) === y[i]) right++;
  return y.length ? right / y.length : 0;
}

// ---- Data for the lesson ----

/** n points on the line y = slope·x + intercept, x uniform in [x0, x1], plus Gaussian noise. */
export function noisyLine(n: number, random: () => number, { slope = 0.6, intercept = 0.4, noise = 0.2, x0 = 0, x1 = 2 } = {}): { x: number[]; y: number[] } {
  const x = Array.from({ length: n }, () => x0 + (x1 - x0) * random()).sort((a, b) => a - b);
  return { x, y: x.map(v => slope * v + intercept + noise * gaussianRandom(random)) };
}

/** The direction (unit vector) from the class-0 blob's centre to the class-1 blob's. */
export const BLOB_AXIS: [number, number] = [Math.cos(0.5), Math.sin(0.5)];

/**
 * Two elongated Gaussian blobs, n/2 points each, with centres ∓separation/2 along
 * BLOB_AXIS. The standard deviation is `spread` along the axis and `stretch`·spread across
 * it (along the boundary between them). xy is flat (x₀, y₀, x₁, y₁, …).
 */
export function twoBlobs(n: number, separation: number, spread: number, random: () => number, stretch = 2): { xy: number[]; labels: number[] } {
  const xy: number[] = [], labels: number[] = [];
  const [ax, ay] = BLOB_AXIS;
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    const along = (c ? separation / 2 : -separation / 2) + spread * gaussianRandom(random);
    const across = stretch * spread * gaussianRandom(random);
    xy.push(along * ax - across * ay, along * ay + across * ax);
    labels.push(c);
  }
  return { xy, labels };
}

/** k points of one class scattered (uniformly, ±radius) around (cx, cy), e.g. far on the right side of a boundary. */
export function cluster(cx: number, cy: number, k: number, random: () => number, radius = 0.15): number[] {
  const xy: number[] = [];
  for (let i = 0; i < k; i++) xy.push(cx + radius * (2 * random() - 1), cy + radius * (2 * random() - 1));
  return xy;
}

/** The angle of a 2-D boundary's normal w, in degrees (to compare two boundaries). */
export const normalAngle = (m: LinearModel): number => (Math.atan2(m.w[1], m.w[0]) * 180) / Math.PI;
