/**
 * Helpers for maps of MNIST digits (used by the PCA and t-SNE lessons): ten theme-aware
 * colours, a random sample of a split, a fitted plot extent, and how often a point's
 * nearest neighbours in the map are the same digit.
 */
import type { MnistSplit } from '../../deep-learning/shared/mnist';
import { shuffled } from '../../deep-learning/shared/train';
import type { Extent } from '../shared/plot';

/** One colour per digit, built from the theme tokens so both themes stay readable. */
export const DIGIT_COLORS = [
  'var(--series-blue)',
  'var(--series-orange)',
  'var(--series-green)',
  'var(--series-pink)',
  'var(--series-red)',
  'var(--accent-2)',
  'color-mix(in srgb, var(--series-blue) 45%, var(--series-light-green))',
  'var(--warning)',
  'var(--text-muted)',
  'color-mix(in srgb, var(--series-green) 50%, var(--series-orange))',
] as const;

/** `count` random images of `split` (MNIST is stored digit by digit, so never take the first N). */
export function sampleDigits(split: MnistSplit, count: number, random: () => number): { images: Float32Array; labels: Uint8Array; indices: number[] } {
  const indices = shuffled(split.count, random).slice(0, Math.min(count, split.count));
  const images = new Float32Array(indices.length * 784);
  const labels = new Uint8Array(indices.length);
  indices.forEach((r, k) => {
    images.set(split.images.subarray(r * 784, (r + 1) * 784), k * 784);
    labels[k] = split.labels[r];
  });
  return { images, labels, indices };
}

/** A square extent around the (x, y) pairs, with a margin, so the map fills the plot. */
export function fitExtent(xy: ArrayLike<number>, margin = 0.06): Extent {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < xy.length; i += 2) {
    x0 = Math.min(x0, xy[i]); x1 = Math.max(x1, xy[i]);
    y0 = Math.min(y0, xy[i + 1]); y1 = Math.max(y1, xy[i + 1]);
  }
  if (!Number.isFinite(x0)) return { x: [-1, 1], y: [-1, 1] };
  const half = (Math.max(x1 - x0, y1 - y0, 1e-9) / 2) * (1 + 2 * margin);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return { x: [cx - half, cx + half], y: [cy - half, cy + half] };
}

/**
 * For the n points xy (pairs), the fraction of each point's k nearest neighbours in the map
 * that are the same digit: overall and per digit. High means the digit forms its own region.
 */
export function sameDigitNeighbours(xy: ArrayLike<number>, labels: ArrayLike<number>, k = 5): { overall: number; perDigit: number[] } {
  const n = labels.length;
  const same = new Array<number>(10).fill(0), total = new Array<number>(10).fill(0);
  const bestD = new Float64Array(k), bestJ = new Int32Array(k);
  for (let i = 0; i < n; i++) {
    bestD.fill(Infinity);
    const xi = xy[2 * i], yi = xy[2 * i + 1];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = (xy[2 * j] - xi) ** 2 + (xy[2 * j + 1] - yi) ** 2;
      if (d >= bestD[k - 1]) continue;
      // Insert into the sorted list of the k best so far.
      let a = k - 1;
      while (a > 0 && bestD[a - 1] > d) { bestD[a] = bestD[a - 1]; bestJ[a] = bestJ[a - 1]; a--; }
      bestD[a] = d; bestJ[a] = j;
    }
    const c = labels[i];
    for (let a = 0; a < k; a++) if (labels[bestJ[a]] === c) same[c]++;
    total[c] += k;
  }
  const all = same.reduce((s, v) => s + v, 0) / Math.max(1, total.reduce((s, v) => s + v, 0));
  return { overall: all, perDigit: same.map((s, c) => (total[c] ? s / total[c] : 0)) };
}
