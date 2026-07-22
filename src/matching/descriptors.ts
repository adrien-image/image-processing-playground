/**
 * Patch descriptors + SSD matching — pure TypeScript.
 * Uses normalised 5×5 patches for illumination-invariant matching.
 */

import type { Keypoint } from './harris';

export interface Match {
  /** Index in the source keypoint array */
  srcIdx: number;
  /** Index in the target keypoint array */
  dstIdx: number;
  /** Sum of Squared Differences (lower = better) */
  distance: number;
}

const PATCH_RADIUS = 2; // 5×5 patch

/**
 * Extract a normalised 5×5 patch descriptor around a keypoint.
 * Descriptor is normalised to zero mean and unit variance.
 */
function extractDescriptor(
  pixels: Float64Array,
  width: number,
  height: number,
  kp: Keypoint,
): Float64Array {
  const patch = new Float64Array((PATCH_RADIUS * 2 + 1) ** 2);
  let idx = 0;
  let sum = 0;

  for (let dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
    for (let dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
      const px = Math.round(kp.x) + dx;
      const py = Math.round(kp.y) + dy;
      if (px >= 0 && px < width && py >= 0 && py < height) {
        patch[idx] = pixels[py * width + px];
      }
      sum += patch[idx];
      idx++;
    }
  }

  // Normalise to zero mean and unit variance
  const mean = sum / patch.length;
  let variance = 0;
  for (let i = 0; i < patch.length; i++) {
    patch[i] -= mean;
    variance += patch[i] * patch[i];
  }
  const std = Math.sqrt(variance / patch.length);
  if (std > 1e-6) {
    for (let i = 0; i < patch.length; i++) {
      patch[i] /= std;
    }
  }

  return patch;
}

/**
 * Compute Sum of Squared Differences between two descriptors.
 */
function ssd(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

/**
 * Match keypoints between two images using SSD on normalised 5×5 patches.
 * Returns mutual-best matches (bidirectional consistency).
 */
export function matchKeypoints(
  pixelsA: Float64Array,
  widthA: number,
  heightA: number,
  keypointsA: Keypoint[],
  pixelsB: Float64Array,
  widthB: number,
  heightB: number,
  keypointsB: Keypoint[],
): Match[] {
  if (keypointsA.length === 0 || keypointsB.length === 0) return [];

  // Extract descriptors for both images
  const descA = keypointsA.map(kp => extractDescriptor(pixelsA, widthA, heightA, kp));
  const descB = keypointsB.map(kp => extractDescriptor(pixelsB, widthB, heightB, kp));

  // For each keypoint in A, find best match in B (forward)
  const forwardMatches: { srcIdx: number; dstIdx: number; dist: number }[] = [];

  for (let i = 0; i < descA.length; i++) {
    let bestDist = Infinity;
    let bestJ = -1;
    let secondBest = Infinity;
    for (let j = 0; j < descB.length; j++) {
      const d = ssd(descA[i], descB[j]);
      if (d < bestDist) {
        secondBest = bestDist;
        bestDist = d;
        bestJ = j;
      } else if (d < secondBest) {
        secondBest = d;
      }
    }
    // Ratio test: best match must be better than second best
    if (bestJ >= 0 && bestDist < secondBest * 0.9) {
      forwardMatches.push({ srcIdx: i, dstIdx: bestJ, dist: bestDist });
    }
  }

  // For each keypoint in B, find best match in A (backward)
  const backwardBest = new Map<number, number>();
  for (let j = 0; j < descB.length; j++) {
    let bestDist = Infinity;
    let bestI = -1;
    for (let i = 0; i < descA.length; i++) {
      const d = ssd(descA[i], descB[j]);
      if (d < bestDist) {
        bestDist = d;
        bestI = i;
      }
    }
    if (bestI >= 0) {
      backwardBest.set(j, bestI);
    }
  }

  // Keep only mutual best matches
  const matches: Match[] = [];
  for (const fwd of forwardMatches) {
    if (backwardBest.get(fwd.dstIdx) === fwd.srcIdx) {
      matches.push({ srcIdx: fwd.srcIdx, dstIdx: fwd.dstIdx, distance: fwd.dist });
    }
  }

  // Sort by distance (best first)
  matches.sort((a, b) => a.distance - b.distance);
  return matches;
}

/**
 * Convenience: get pixel coordinates for a match.
 */
export function matchPoints(
  match: Match,
  keypointsA: Keypoint[],
  keypointsB: Keypoint[],
): [{ x: number; y: number }, { x: number; y: number }] {
  return [
    { x: keypointsA[match.srcIdx].x, y: keypointsA[match.srcIdx].y },
    { x: keypointsB[match.dstIdx].x, y: keypointsB[match.dstIdx].y },
  ];
}
