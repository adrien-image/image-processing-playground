/**
 * t-SNE (t-distributed stochastic neighbour embedding, van der Maaten & Hinton 2008):
 * a 2-D map of high-dimensional points that keeps neighbours together.
 *
 * 1. Affinities in the original space. For each point i, a Gaussian of width σᵢ turns
 *    distances into probabilities of picking j as a neighbour:
 *      p(j|i) = exp(−‖xᵢ − xⱼ‖² / 2σᵢ²) / Σₖ≠ᵢ exp(−‖xᵢ − xₖ‖² / 2σᵢ²)
 *    σᵢ is set by binary search so that the perplexity 2^H(Pᵢ) (H the entropy in bits)
 *    equals a chosen value: roughly the number of neighbours each point "sees". Dense
 *    regions get small σ, sparse ones large σ. The joint probabilities are symmetrised:
 *      pᵢⱼ = (p(j|i) + p(i|j)) / 2n,      Σᵢⱼ pᵢⱼ = 1
 * 2. Affinities in the map use a Student t (Cauchy) kernel, with heavy tails:
 *      qᵢⱼ = (1 + ‖yᵢ − yⱼ‖²)⁻¹ / Σₖ≠ₗ (1 + ‖yₖ − yₗ‖²)⁻¹
 * 3. Gradient descent on KL(P ‖ Q) = Σ pᵢⱼ log(pᵢⱼ / qᵢⱼ), whose gradient is
 *      ∂KL/∂yᵢ = 4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)(1 + ‖yᵢ − yⱼ‖²)⁻¹
 *    with the usual tricks: early exaggeration (P × 12 for the first 250 iterations, so
 *    clusters form before they settle), momentum 0.5 then 0.8, and per-coordinate gains
 *    (grow the step while the gradient keeps its sign, shrink it when it flips), with a
 *    step size that grows with n.
 * This is the exact O(n²) algorithm: fine for about a thousand points (Barnes–Hut or
 * FFT approximations are what make t-SNE scale to millions).
 *
 * Matrices are row-major Float64Arrays; the embedding Y is n × 2: (Y[2i], Y[2i + 1]).
 */
import { gaussianRandom } from '../../lib/random';

/** All squared Euclidean distances between the n rows (length d) of X: n × n. */
export function squaredDistances(X: ArrayLike<number>, n: number, d: number): Float64Array {
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    let s = 0;
    for (let c = 0; c < d; c++) {
      const t = X[i * d + c] - X[j * d + c];
      s += t * t;
    }
    D[i * n + j] = s;
    D[j * n + i] = s;
  }
  return D;
}

/**
 * Row i of the conditional probabilities p(j|i) for precision β = 1/2σ², written into
 * `row`; returns the entropy H in nats (perplexity = e^H = 2^H_bits).
 */
function gaussianRow(D: Float64Array, n: number, i: number, beta: number, row: Float64Array): number {
  // Subtract the smallest distance first so exp() never underflows to 0 for every j.
  let dmin = Infinity;
  for (let j = 0; j < n; j++) if (j !== i && D[i * n + j] < dmin) dmin = D[i * n + j];
  let sum = 0, weighted = 0;
  for (let j = 0; j < n; j++) {
    if (j === i) { row[j] = 0; continue; }
    const dj = D[i * n + j] - dmin;
    const e = Math.exp(-beta * dj);
    row[j] = e;
    sum += e;
    weighted += dj * e;
  }
  for (let j = 0; j < n; j++) row[j] /= sum;
  // H = −Σ p log p = log Σe + β·E[d − dmin]
  return Math.log(sum) + (beta * weighted) / sum;
}

/**
 * The conditional probabilities p(j|i) (row i sums to 1), with each σᵢ found by a binary
 * search on β = 1/2σᵢ² until the row's perplexity matches `perplexity`.
 */
export function conditionalP(D: Float64Array, n: number, perplexity: number, tolerance = 1e-5): { P: Float64Array; sigma: Float64Array } {
  const P = new Float64Array(n * n);
  const sigma = new Float64Array(n);
  const target = Math.log(perplexity);
  for (let i = 0; i < n; i++) {
    const row = P.subarray(i * n, (i + 1) * n);
    let beta = 1, lo = 0, hi = Infinity;
    for (let it = 0; it < 100; it++) {
      const H = gaussianRow(D, n, i, beta, row);
      if (Math.abs(H - target) < tolerance) break;
      // A larger β (narrower Gaussian) means fewer neighbours, a lower entropy.
      if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; }
      else { hi = beta; beta = (beta + lo) / 2; }
    }
    sigma[i] = Math.sqrt(1 / (2 * beta));
  }
  return { P, sigma };
}

/** The perplexity e^H of row i of a row-stochastic matrix. */
export function rowPerplexity(P: Float64Array, n: number, i: number): number {
  let H = 0;
  for (let j = 0; j < n; j++) {
    const p = P[i * n + j];
    if (p > 0) H -= p * Math.log(p);
  }
  return Math.exp(H);
}

/** Symmetrised joint probabilities pᵢⱼ = (p(j|i) + p(i|j)) / 2n: symmetric, summing to 1. */
export function jointP(conditional: Float64Array, n: number): Float64Array {
  const P = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    P[i * n + j] = Math.max((conditional[i * n + j] + conditional[j * n + i]) / (2 * n), 1e-12);
  }
  for (let i = 0; i < n; i++) P[i * n + i] = 0;
  return P;
}

/** Affinities P for the n × d points X at a given perplexity. */
export function affinities(X: ArrayLike<number>, n: number, d: number, perplexity: number): Float64Array {
  return jointP(conditionalP(squaredDistances(X, n, d), n, perplexity).P, n);
}

/**
 * KL(P ‖ Q) and its gradient with respect to the n × 2 embedding Y. `exaggeration`
 * multiplies P in the gradient (not in the returned KL, which is always the true one).
 */
export function klGradient(P: Float64Array, Y: Float64Array, n: number, exaggeration = 1): { kl: number; grad: Float64Array } {
  // With wᵢⱼ = (1 + ‖yᵢ − yⱼ‖²)⁻¹ and Z = Σ wᵢⱼ, qᵢⱼ = wᵢⱼ / Z and
  //   (pᵢⱼ − qᵢⱼ)·wᵢⱼ = pᵢⱼ·wᵢⱼ − wᵢⱼ² / Z,
  // so one pass over the pairs collects an attractive sum (P-weighted) and a repulsive
  // one (divided by Z at the end), and KL = Σ p log(p / w) + log Z · Σ p.
  const attract = new Float64Array(2 * n), repel = new Float64Array(2 * n);
  let Z = 0, plogpw = 0, psum = 0;
  for (let i = 0; i < n; i++) {
    const xi = Y[2 * i], yi = Y[2 * i + 1];
    let ax = 0, ay = 0, rx = 0, ry = 0;
    for (let j = i + 1; j < n; j++) {
      const dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
      const w = 1 / (1 + dx * dx + dy * dy);
      const p = P[i * n + j];
      Z += 2 * w;
      const a = p * w, r = w * w;
      ax += a * dx; ay += a * dy; rx += r * dx; ry += r * dy;
      attract[2 * j] -= a * dx; attract[2 * j + 1] -= a * dy;
      repel[2 * j] -= r * dx; repel[2 * j + 1] -= r * dy;
      if (p > 0) { plogpw += 2 * p * Math.log(p / w); psum += 2 * p; }
    }
    attract[2 * i] += ax; attract[2 * i + 1] += ay;
    repel[2 * i] += rx; repel[2 * i + 1] += ry;
  }
  const grad = new Float64Array(2 * n);
  for (let c = 0; c < 2 * n; c++) grad[c] = 4 * (exaggeration * attract[c] - repel[c] / Z);
  return { kl: plogpw + Math.log(Z) * psum, grad };
}

export interface TsneOptions {
  random: () => number;
  /**
   * Step size (default max(n/4, 50): 200 for 800 points). Each pᵢⱼ is about 1/n, so the
   * gradient shrinks as n grows and the step has to grow with it; too large a step for a
   * small set throws points out of their clusters.
   */
  learningRate?: number;
  /** Early exaggeration factor and how many iterations it lasts. */
  exaggeration?: number;
  exaggerationIterations?: number;
}

/** A running t-SNE optimisation. */
export class Tsne {
  readonly n: number;
  readonly P: Float64Array;
  /** The n × 2 embedding. */
  Y: Float64Array;
  iteration = 0;
  /** KL(P ‖ Q) at the start of the last step. */
  kl = NaN;
  private velocity: Float64Array;
  private gains: Float64Array;
  private readonly learningRate: number;
  private readonly exaggeration: number;
  private readonly exaggerationIterations: number;

  constructor(P: Float64Array, n: number, { random, learningRate = Math.max(n / 4, 50), exaggeration = 12, exaggerationIterations = 250 }: TsneOptions) {
    this.n = n;
    this.P = P;
    this.learningRate = learningRate;
    this.exaggeration = exaggeration;
    this.exaggerationIterations = exaggerationIterations;
    // Start all points almost on top of each other: tiny random positions.
    this.Y = Float64Array.from({ length: 2 * n }, () => 1e-4 * gaussianRandom(random));
    this.velocity = new Float64Array(2 * n);
    this.gains = new Float64Array(2 * n).fill(1);
  }

  /** In the early-exaggeration phase? */
  get exaggerating(): boolean {
    return this.iteration < this.exaggerationIterations;
  }

  /** One gradient step with momentum and gains. */
  step(): void {
    const { n, Y, velocity, gains } = this;
    const early = this.exaggerating;
    const { kl, grad } = klGradient(this.P, Y, n, early ? this.exaggeration : 1);
    const momentum = early ? 0.5 : 0.8;
    for (let c = 0; c < 2 * n; c++) {
      // Gradient pointing the way we are already moving: we overshot, so shrink the gain.
      gains[c] = Math.sign(grad[c]) !== Math.sign(velocity[c]) ? gains[c] + 0.2 : Math.max(gains[c] * 0.8, 0.01);
      velocity[c] = momentum * velocity[c] - this.learningRate * gains[c] * grad[c];
      Y[c] += velocity[c];
    }
    // Keep the map centred on the origin (KL does not care where it is).
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += Y[2 * i]; my += Y[2 * i + 1]; }
    mx /= n; my /= n;
    for (let i = 0; i < n; i++) { Y[2 * i] -= mx; Y[2 * i + 1] -= my; }
    this.kl = kl;
    this.iteration++;
  }
}

/** The k nearest neighbours of every point (by squared distance in D, n × n), excluding itself: n × k. */
export function nearestNeighbours(D: Float64Array, n: number, k: number): Int32Array {
  const out = new Int32Array(n * k);
  const bestD = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    bestD.fill(Infinity);
    const best = out.subarray(i * k, (i + 1) * k);
    for (let j = 0; j < n; j++) {
      const d = D[i * n + j];
      if (j === i || d >= bestD[k - 1]) continue;
      // Insert j into the sorted list of the k nearest so far.
      let a = k - 1;
      while (a > 0 && bestD[a - 1] > d) { bestD[a] = bestD[a - 1]; best[a] = best[a - 1]; a--; }
      bestD[a] = d;
      best[a] = j;
    }
  }
  return out;
}

/** Of every point's k nearest neighbours in the n × dims points X, the fraction with the same label. */
export function knnAgreement(X: ArrayLike<number>, dims: number, labels: ArrayLike<number>, n: number, k = 5): number {
  const nn = nearestNeighbours(squaredDistances(X, n, dims), n, k);
  let same = 0;
  for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) if (labels[nn[i * k + a]] === labels[i]) same++;
  return same / (n * k);
}

/**
 * How well a map keeps neighbourhoods: of every point's k nearest neighbours in the original
 * space (`highNeighbours`, from nearestNeighbours), the fraction that are also among its k
 * nearest in the n × dims map Y.
 */
export function neighbourhoodKept(highNeighbours: Int32Array, Y: ArrayLike<number>, dims: number, n: number, k: number): number {
  const low = nearestNeighbours(squaredDistances(Y, n, dims), n, k);
  let kept = 0;
  for (let i = 0; i < n; i++) {
    const mine = new Set(low.subarray(i * k, (i + 1) * k));
    for (let a = 0; a < k; a++) if (mine.has(highNeighbours[i * k + a])) kept++;
  }
  return kept / (n * k);
}
