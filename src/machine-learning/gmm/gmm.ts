/**
 * Gaussian mixture models in 2-D, fitted by expectation–maximisation (EM), next to
 * k-means (Lloyd's algorithm) on the same data.
 *
 * The generative model: pick a component k with probability πₖ, then draw x from the
 * Gaussian N(μₖ, Σₖ). Each component has its own weight, centre and full 2 × 2
 * covariance, so clusters can be stretched, tilted and of different sizes.
 *
 * EM alternates two steps, and neither can lower the log-likelihood Σᵢ log p(xᵢ):
 *   E-step  rᵢₖ = πₖ N(xᵢ | μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ | μⱼ, Σⱼ)       (Bayes: who made xᵢ?)
 *   M-step  Nₖ = Σᵢ rᵢₖ,  πₖ = Nₖ / n,  μₖ = Σᵢ rᵢₖ xᵢ / Nₖ,
 *           Σₖ = (Σᵢ rᵢₖ (xᵢ − μₖ)(xᵢ − μₖ)ᵀ + λI) / Nₖ
 * The λI is a small covariance floor: without it a component can shrink onto a single
 * point, where its density, and the likelihood, go to infinity. It is exactly the
 * M-step of the likelihood minus a penalty (λ/2) Σₖ tr(Σₖ⁻¹), so EM never decreases
 * that penalised objective, which is what `objective` returns.
 *
 * k-means is the limit where every Σₖ = σ²I with σ → 0 and equal weights: the
 * responsibilities become 0 or 1 (nearest centre) and the M-step is a plain mean.
 */
import { gaussianRandom } from '../../lib/random';
import type { Points2D } from '../shared/datasets';

/** One component: weight π, mean μ and covariance Σ = [σxx, σxy, σyy]. */
export interface Component {
  weight: number;
  mean: [number, number];
  cov: [number, number, number];
}

export type Mixture = Component[];
export type Point = [number, number];

/** The covariance floor λ (data live in about [−1, 1]², so a collapsed component keeps σ ≈ √λ ≈ 0.03). */
export const REG = 1e-3;

/** Label of a point the learner added by clicking: it belongs to no true cluster. */
export const ADDED = 255;

const LOG_2PI = Math.log(2 * Math.PI);

/** log N(x, y | μ, Σ) for one component. */
export function logGaussian(x: number, y: number, { mean, cov }: Component): number {
  const [a, b, c] = cov;
  const det = a * c - b * b;
  const dx = x - mean[0], dy = y - mean[1];
  // (x − μ)ᵀ Σ⁻¹ (x − μ), with Σ⁻¹ = [c, −b; −b, a] / det.
  const q = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
  return -LOG_2PI - 0.5 * Math.log(det) - 0.5 * q;
}

/** The responsibilities of every component for (x, y), and log p(x, y), via log-sum-exp. */
export function posterior(x: number, y: number, mix: Mixture, out = new Float64Array(mix.length)): { resp: Float64Array; logP: number } {
  let top = -Infinity;
  for (let k = 0; k < mix.length; k++) {
    out[k] = Math.log(Math.max(mix[k].weight, 1e-300)) + logGaussian(x, y, mix[k]);
    top = Math.max(top, out[k]);
  }
  let sum = 0;
  for (let k = 0; k < mix.length; k++) { out[k] = Math.exp(out[k] - top); sum += out[k]; }
  for (let k = 0; k < mix.length; k++) out[k] /= sum;
  return { resp: out, logP: top + Math.log(sum) };
}

/** E-step: resp[i · K + k] = rᵢₖ, and the log-likelihood of the current parameters. */
export function eStep(xy: Float32Array, mix: Mixture): { resp: Float64Array; logLik: number } {
  const n = xy.length / 2, K = mix.length;
  const resp = new Float64Array(n * K);
  let logLik = 0;
  for (let i = 0; i < n; i++) logLik += posterior(xy[2 * i], xy[2 * i + 1], mix, resp.subarray(i * K, (i + 1) * K)).logP;
  return { resp, logLik };
}

/** M-step: the weighted weights, means and covariances (with the floor λ). */
export function mStep(xy: Float32Array, resp: Float64Array, K: number, reg = REG): Mixture {
  const n = xy.length / 2;
  const mix: Mixture = [];
  for (let k = 0; k < K; k++) {
    let nk = 0, sx = 0, sy = 0;
    for (let i = 0; i < n; i++) {
      const r = resp[i * K + k];
      nk += r; sx += r * xy[2 * i]; sy += r * xy[2 * i + 1];
    }
    nk = Math.max(nk, 1e-10); // a component nobody claims: keep it finite
    const mx = sx / nk, my = sy / nk;
    let a = reg, b = 0, c = reg;
    for (let i = 0; i < n; i++) {
      const r = resp[i * K + k], dx = xy[2 * i] - mx, dy = xy[2 * i + 1] - my;
      a += r * dx * dx; b += r * dx * dy; c += r * dy * dy;
    }
    mix.push({ weight: nk / n, mean: [mx, my], cov: [a / nk, b / nk, c / nk] });
  }
  return mix;
}

/** The penalty (λ/2) Σₖ tr(Σₖ⁻¹) that the covariance floor corresponds to. */
export function penalty(mix: Mixture, reg = REG): number {
  let s = 0;
  for (const { cov: [a, b, c] } of mix) s += (a + c) / (a * c - b * b);
  return (reg / 2) * s;
}

/** What EM climbs: the log-likelihood minus the floor's penalty. */
export const objective = (xy: Float32Array, mix: Mixture, reg = REG) => eStep(xy, mix).logLik - penalty(mix, reg);

/**
 * Starting centres: `kmeans++` picks the first at random and each next one with
 * probability proportional to its squared distance to the nearest centre so far, so
 * they tend to spread over the clusters; `random` picks K distinct points.
 */
export function initialCentres(xy: Float32Array, K: number, random: () => number, method: 'kmeans++' | 'random' = 'kmeans++'): Point[] {
  const n = xy.length / 2;
  const at = (i: number): Point => [xy[2 * i], xy[2 * i + 1]];
  if (method === 'random') {
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < Math.min(K, n); i++) {
      const j = i + Math.floor(random() * (n - i));
      [order[i], order[j]] = [order[j], order[i]];
    }
    return order.slice(0, K).map(at);
  }
  const centres = [at(Math.floor(random() * n))];
  const d2 = new Float64Array(n).fill(Infinity);
  while (centres.length < K) {
    const [cx, cy] = centres[centres.length - 1];
    let total = 0;
    for (let i = 0; i < n; i++) {
      d2[i] = Math.min(d2[i], (xy[2 * i] - cx) ** 2 + (xy[2 * i + 1] - cy) ** 2);
      total += d2[i];
    }
    let u = random() * total, pick = n - 1;
    for (let i = 0; i < n; i++) { u -= d2[i]; if (u <= 0) { pick = i; break; } }
    centres.push(at(pick));
  }
  return centres;
}

/** A mixture centred on the given points, with equal weights and round covariances of the data's spread / K. */
export function startMixture(xy: Float32Array, centres: Point[]): Mixture {
  const n = xy.length / 2, K = centres.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xy[2 * i]; my += xy[2 * i + 1]; }
  mx /= n; my /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (xy[2 * i] - mx) ** 2 + (xy[2 * i + 1] - my) ** 2;
  const s2 = Math.max(v / (2 * n) / K, REG);
  return centres.map(([x, y]) => ({ weight: 1 / K, mean: [x, y], cov: [s2, 0, s2] }));
}

/** Run `iterations` full EM steps; history[t] is the objective after t steps. */
export function fitGmm(xy: Float32Array, mix: Mixture, iterations: number, reg = REG): { mix: Mixture; history: number[] } {
  const history = [eStep(xy, mix).logLik - penalty(mix, reg)];
  for (let t = 0; t < iterations; t++) {
    mix = mStep(xy, eStep(xy, mix).resp, mix.length, reg);
    history.push(eStep(xy, mix).logLik - penalty(mix, reg));
    if (Math.abs(history[t + 1] - history[t]) < 1e-9 * Math.abs(history[t])) break;
  }
  return { mix, history };
}

/** The most responsible component of every point. */
export function hardLabels(resp: Float64Array, K: number): Uint8Array {
  const n = resp.length / K;
  return Uint8Array.from({ length: n }, (_, i) => {
    let best = 0;
    for (let k = 1; k < K; k++) if (resp[i * K + k] > resp[i * K + best]) best = k;
    return best;
  });
}

// --- k-means (Lloyd), from the same starting centres -------------------------------------

/** Assignment step: the index of every point's nearest centre. */
export function assignNearest(xy: Float32Array, centres: Point[]): Uint8Array {
  const n = xy.length / 2;
  return Uint8Array.from({ length: n }, (_, i) => nearest(xy[2 * i], xy[2 * i + 1], centres));
}

export function nearest(x: number, y: number, centres: Point[]): number {
  let best = 0, bestD = Infinity;
  centres.forEach(([cx, cy], k) => {
    const d = (x - cx) ** 2 + (y - cy) ** 2;
    if (d < bestD) { bestD = d; best = k; }
  });
  return best;
}

/** Update step: every centre moves to the mean of its points (an empty cluster stays put). */
export function updateCentres(xy: Float32Array, labels: Uint8Array, centres: Point[]): Point[] {
  const sums = centres.map(() => [0, 0, 0]);
  labels.forEach((k, i) => { sums[k][0] += xy[2 * i]; sums[k][1] += xy[2 * i + 1]; sums[k][2]++; });
  return centres.map((c, k) => (sums[k][2] ? [sums[k][0] / sums[k][2], sums[k][1] / sums[k][2]] : c));
}

/** k-means' objective: the sum of squared distances to the assigned centres. */
export function inertia(xy: Float32Array, labels: Uint8Array, centres: Point[]): number {
  let s = 0;
  labels.forEach((k, i) => { s += (xy[2 * i] - centres[k][0]) ** 2 + (xy[2 * i + 1] - centres[k][1]) ** 2; });
  return s;
}

// --- Choosing K ---------------------------------------------------------------------------

/** Free parameters of a 2-D, K-component, full-covariance mixture: K means (2), covariances (3), K − 1 weights. */
export const parameterCount = (K: number) => 6 * K - 1;

/** Bayesian information criterion, −2 log L + p ln n: lower is better. */
export function bic(xy: Float32Array, mix: Mixture): number {
  const n = xy.length / 2;
  return -2 * eStep(xy, mix).logLik + parameterCount(mix.length) * Math.log(n);
}

/** BIC for K = 1 … kMax, each from the best of a few k-means++ starts. */
export function bicCurve(xy: Float32Array, kMax: number, random: () => number, restarts = 3, iterations = 150): number[] {
  return Array.from({ length: kMax }, (_, j) => {
    let best = Infinity;
    for (let r = 0; r < (j === 0 ? 1 : restarts); r++) {
      const { mix } = fitGmm(xy, startMixture(xy, initialCentres(xy, j + 1, random)), iterations);
      best = Math.min(best, bic(xy, mix));
    }
    return best;
  });
}

// --- Data ---------------------------------------------------------------------------------

export type MixtureData = 'stretched' | 'unequal' | 'round';

export const DATA_LABELS: Record<MixtureData, string> = {
  stretched: 'Stretched clusters', unequal: 'Unequal sizes', round: 'Round blobs',
};

/** The true mixture behind each dataset: three components in about [−1.2, 1.2]². */
export function trueMixture(kind: MixtureData): Mixture {
  if (kind === 'stretched') {
    // Three long, thin, parallel clusters tilted by 35°, side by side.
    const t = (35 * Math.PI) / 180, u: Point = [Math.cos(t), Math.sin(t)], v: Point = [-u[1], u[0]];
    const along = 0.42 ** 2, across = 0.05 ** 2;
    const cov: [number, number, number] = [
      along * u[0] * u[0] + across * v[0] * v[0], along * u[0] * u[1] + across * v[0] * v[1], along * u[1] * u[1] + across * v[1] * v[1],
    ];
    return [-1, 0, 1].map(s => ({ weight: 1 / 3, mean: [0.38 * s * v[0], 0.38 * s * v[1]], cov }));
  }
  if (kind === 'unequal') {
    // One big, wide cluster and two small, tight ones beside it.
    return [
      { weight: 0.7, mean: [-0.3, -0.05], cov: [0.16, 0.02, 0.12] },
      { weight: 0.15, mean: [0.62, 0.5], cov: [0.006, 0, 0.006] },
      { weight: 0.15, mean: [0.66, -0.42], cov: [0.006, 0, 0.006] },
    ];
  }
  return [
    { weight: 1 / 3, mean: [-0.6, 0.45], cov: [0.02, 0, 0.02] },
    { weight: 1 / 3, mean: [0.6, 0.5], cov: [0.02, 0, 0.02] },
    { weight: 1 / 3, mean: [0, -0.6], cov: [0.02, 0, 0.02] },
  ];
}

/** n points drawn from a mixture; labels hold the component each came from (exact counts nπₖ). */
export function sampleMixture(mix: Mixture, n: number, random: () => number): Points2D {
  const xy = new Float32Array(n * 2), labels = new Uint8Array(n);
  const counts = mix.map(c => Math.round(c.weight * n));
  counts[0] += n - counts.reduce((a, b) => a + b, 0);
  let i = 0;
  mix.forEach(({ mean, cov: [a, b, c] }, k) => {
    // x = μ + L z with L the Cholesky factor of Σ.
    const l11 = Math.sqrt(a), l21 = b / l11, l22 = Math.sqrt(c - l21 * l21);
    for (let j = 0; j < counts[k]; j++, i++) {
      const z1 = gaussianRandom(random), z2 = gaussianRandom(random);
      xy[2 * i] = mean[0] + l11 * z1;
      xy[2 * i + 1] = mean[1] + l21 * z1 + l22 * z2;
      labels[i] = k;
    }
  });
  return { xy, labels, n, classes: mix.length };
}

export const makeMixtureData = (kind: MixtureData, n: number, random: () => number) => sampleMixture(trueMixture(kind), n, random);

/**
 * How well predicted clusters match the true ones, up to relabelling: the best accuracy
 * over all permutations of K labels (K ≤ 6). Points labelled ADDED are ignored.
 */
export function agreement(pred: ArrayLike<number>, truth: ArrayLike<number>, K: number): number {
  const counts = new Float64Array(K * K);
  let total = 0;
  for (let i = 0; i < truth.length; i++) {
    if (truth[i] === ADDED || truth[i] >= K) { if (truth[i] !== ADDED) total++; continue; }
    counts[pred[i] * K + truth[i]]++;
    total++;
  }
  let best = 0;
  const perm = Array.from({ length: K }, (_, i) => i);
  const visit = (at: number) => {
    if (at === K) {
      let s = 0;
      for (let k = 0; k < K; k++) s += counts[k * K + perm[k]];
      best = Math.max(best, s);
      return;
    }
    for (let j = at; j < K; j++) {
      [perm[at], perm[j]] = [perm[j], perm[at]];
      visit(at + 1);
      [perm[at], perm[j]] = [perm[j], perm[at]];
    }
  };
  visit(0);
  return total ? best / total : 0;
}
