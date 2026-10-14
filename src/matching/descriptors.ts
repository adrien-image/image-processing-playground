/**
 * Rotation-normalised patch descriptors + SSD matching — pure TypeScript.
 *
 * Each keypoint gets a dominant orientation (the average gradient direction around
 * it); its patch is then sampled in that rotated frame, so the same corner seen
 * rotated in another image produces (nearly) the same descriptor.
 */

import type { Keypoint } from './harris';

export interface Match {
  /** Index in the source keypoint array */
  srcIdx: number;
  /** Index in the target keypoint array */
  dstIdx: number;
  /** Sum of Squared Differences between descriptors (lower = better) */
  distance: number;
}

/** Descriptor patch is (2·PATCH_RADIUS + 1)² samples. */
const PATCH_RADIUS = 4;
/** Radius of the window used to estimate orientation. */
const ORIENTATION_RADIUS = 6;
/**
 * Lowe's ratio test: keep a match only if the best candidate is clearly closer
 * than the runner-up. Applied to Euclidean distances (not squared SSD).
 */
export const RATIO = 0.8;

function bilinear(pixels: Float64Array, width: number, height: number, x: number, y: number): number {
  const cx = Math.min(Math.max(x, 0), width - 1);
  const cy = Math.min(Math.max(y, 0), height - 1);
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1);
  const fx = cx - x0, fy = cy - y0;
  const top = pixels[y0 * width + x0] * (1 - fx) + pixels[y0 * width + x1] * fx;
  const bottom = pixels[y1 * width + x0] * (1 - fx) + pixels[y1 * width + x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

/** Dominant gradient direction around a keypoint (Gaussian-weighted mean gradient). */
export function keypointOrientation(pixels: Float64Array, width: number, height: number, kp: Keypoint): number {
  let sx = 0, sy = 0;
  const sigma = ORIENTATION_RADIUS / 2;
  for (let dy = -ORIENTATION_RADIUS; dy <= ORIENTATION_RADIUS; dy++) {
    for (let dx = -ORIENTATION_RADIUS; dx <= ORIENTATION_RADIUS; dx++) {
      const x = kp.x + dx, y = kp.y + dy;
      const gx = bilinear(pixels, width, height, x + 1, y) - bilinear(pixels, width, height, x - 1, y);
      const gy = bilinear(pixels, width, height, x, y + 1) - bilinear(pixels, width, height, x, y - 1);
      const w = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      sx += gx * w;
      sy += gy * w;
    }
  }
  return Math.atan2(sy, sx);
}

/** Sample a patch in the keypoint's rotated frame and normalise it to zero mean, unit variance. */
export function extractDescriptor(pixels: Float64Array, width: number, height: number, kp: Keypoint): Float64Array {
  const angle = keypointOrientation(pixels, width, height, kp);
  const c = Math.cos(angle), s = Math.sin(angle);
  const size = PATCH_RADIUS * 2 + 1;
  const patch = new Float64Array(size * size);
  let idx = 0, sum = 0;
  for (let v = -PATCH_RADIUS; v <= PATCH_RADIUS; v++) {
    for (let u = -PATCH_RADIUS; u <= PATCH_RADIUS; u++) {
      // (u, v) in the keypoint frame → image coordinates
      const x = kp.x + u * c - v * s;
      const y = kp.y + u * s + v * c;
      patch[idx] = bilinear(pixels, width, height, x, y);
      sum += patch[idx++];
    }
  }

  const mean = sum / patch.length;
  let variance = 0;
  for (let i = 0; i < patch.length; i++) {
    patch[i] -= mean;
    variance += patch[i] * patch[i];
  }
  const std = Math.sqrt(variance / patch.length);
  if (std > 1e-6) for (let i = 0; i < patch.length; i++) patch[i] /= std;
  return patch;
}

/** Sum of Squared Differences between two descriptors. */
function ssd(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

/**
 * Match keypoints between two images.
 * Keeps a pair only if it passes the ratio test AND is a mutual best match
 * (A's best in B is j, and B's best in A is i).
 */
export function matchKeypoints(
  pixelsA: Float64Array, widthA: number, heightA: number, keypointsA: Keypoint[],
  pixelsB: Float64Array, widthB: number, heightB: number, keypointsB: Keypoint[],
): Match[] {
  if (keypointsA.length === 0 || keypointsB.length === 0) return [];

  const descA = keypointsA.map(kp => extractDescriptor(pixelsA, widthA, heightA, kp));
  const descB = keypointsB.map(kp => extractDescriptor(pixelsB, widthB, heightB, kp));
  const dist = descA.map(a => descB.map(b => ssd(a, b)));

  // Ratio test on distances: √best < RATIO·√second  ⇔  best < RATIO²·second (SSD is squared)
  const ratioSq = RATIO * RATIO;
  const matches: Match[] = [];
  for (let i = 0; i < descA.length; i++) {
    let best = Infinity, second = Infinity, bestJ = -1;
    for (let j = 0; j < descB.length; j++) {
      const d = dist[i][j];
      if (d < best) { second = best; best = d; bestJ = j; }
      else if (d < second) second = d;
    }
    if (bestJ < 0 || !(best < ratioSq * second)) continue;

    // Mutual check: is i also B[bestJ]'s best match in A?
    let backBest = Infinity, backI = -1;
    for (let i2 = 0; i2 < descA.length; i2++) {
      if (dist[i2][bestJ] < backBest) { backBest = dist[i2][bestJ]; backI = i2; }
    }
    if (backI === i) matches.push({ srcIdx: i, dstIdx: bestJ, distance: best });
  }

  matches.sort((a, b) => a.distance - b.distance);
  return matches;
}
