/**
 * Small synthetic datasets for the machine-learning lessons, all seeded so a demo
 * looks the same every time it loads.
 *
 * 2-D points live in roughly [−1, 1]²: `xy` holds x₀, y₀, x₁, y₁, … and `labels` the
 * class of each point.
 */
import { gaussianRandom } from '../../lib/random';

export interface Points2D {
  xy: Float32Array;
  labels: Uint8Array;
  n: number;
  classes: number;
}

export type Shape2D = 'blobs' | 'moons' | 'circles' | 'xor' | 'spiral';

export const SHAPE_LABELS: Record<Shape2D, string> = {
  blobs: 'Blobs', moons: 'Two moons', circles: 'Circles', xor: 'XOR', spiral: 'Spiral',
};

/**
 * n points of a classic 2-D classification problem, with Gaussian noise of standard
 * deviation `noise` added to every coordinate. `classes` only affects blobs and spiral
 * (the others always have two).
 */
export function makeClassification(shape: Shape2D, n: number, noise: number, random: () => number, classes = 2): Points2D {
  const k = shape === 'blobs' || shape === 'spiral' ? classes : 2;
  const xy = new Float32Array(n * 2);
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % k;
    let x = 0, y = 0;
    if (shape === 'blobs') {
      const a = (2 * Math.PI * c) / k + 0.4;
      x = 0.55 * Math.cos(a) + 0.18 * gaussianRandom(random);
      y = 0.55 * Math.sin(a) + 0.18 * gaussianRandom(random);
    } else if (shape === 'moons') {
      const t = Math.PI * random();
      x = c === 0 ? Math.cos(t) - 0.5 : 0.5 - Math.cos(t);
      y = c === 0 ? Math.sin(t) - 0.25 : 0.25 - Math.sin(t);
      x *= 0.75; y *= 0.75;
    } else if (shape === 'circles') {
      const t = 2 * Math.PI * random(), r = c === 0 ? 0.35 : 0.85;
      x = r * Math.cos(t); y = r * Math.sin(t);
    } else if (shape === 'xor') {
      x = (random() * 0.8 + 0.1) * (random() < 0.5 ? -1 : 1);
      y = (random() * 0.8 + 0.1) * (random() < 0.5 ? -1 : 1);
      labels[i] = x * y > 0 ? 0 : 1;
    } else {
      const t = random();
      const a = 3.5 * Math.PI * t + (2 * Math.PI * c) / k;
      x = 0.95 * t * Math.cos(a); y = 0.95 * t * Math.sin(a);
    }
    xy[2 * i] = x + noise * gaussianRandom(random);
    xy[2 * i + 1] = y + noise * gaussianRandom(random);
    if (shape !== 'xor') labels[i] = c;
  }
  return { xy, labels, n, classes: k };
}

export interface Series1D {
  x: Float32Array;
  y: Float32Array;
  n: number;
}

/** The smooth curve behind makeRegression: a sine with a slow trend, on x ∈ [−1, 1]. */
export const trueCurve = (x: number): number => Math.sin(2.6 * x) * 0.6 + 0.25 * x;

/** n points (x, f(x) + noise) with x uniform in [−1, 1], sorted by x. */
export function makeRegression(n: number, noise: number, random: () => number, f = trueCurve): Series1D {
  const xs = Array.from({ length: n }, () => random() * 2 - 1).sort((a, b) => a - b);
  const x = Float32Array.from(xs);
  const y = Float32Array.from(xs, v => f(v) + noise * gaussianRandom(random));
  return { x, y, n };
}

/** The rows of a 2-D dataset at the given indices (e.g. a train / test split). */
export function subset(data: Points2D, rows: readonly number[]): Points2D {
  const xy = new Float32Array(rows.length * 2);
  const labels = new Uint8Array(rows.length);
  rows.forEach((r, i) => { xy[2 * i] = data.xy[2 * r]; xy[2 * i + 1] = data.xy[2 * r + 1]; labels[i] = data.labels[r]; });
  return { xy, labels, n: rows.length, classes: data.classes };
}

/** Fraction of `predicted` equal to `labels`. */
export function accuracyOf(predicted: ArrayLike<number>, labels: ArrayLike<number>): number {
  let right = 0;
  for (let i = 0; i < labels.length; i++) if (predicted[i] === labels[i]) right++;
  return labels.length ? right / labels.length : 0;
}
