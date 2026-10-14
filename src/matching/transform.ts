/**
 * Similarity transform estimation: rotation θ, uniform scale s and translation (tx, ty),
 *
 *   x' = s·cos θ · x − s·sin θ · y + tx
 *   y' = s·sin θ · x + s·cos θ · y + ty
 *
 * `estimateSimilarity` is a closed-form least-squares fit; `ransacSimilarity` wraps it in
 * RANSAC so a few wrong matches (outliers) cannot drag the estimate away.
 */

import { seededRandom } from '../lib/random';

export interface Point { x: number; y: number; }

export interface Similarity {
  /** Rotation in radians */
  angle: number;
  /** Uniform scale factor */
  scale: number;
  tx: number;
  ty: number;
}

/**
 * Least-squares similarity from ≥ 2 correspondences (Umeyama / Procrustes in 2D).
 * Working on centroid-demeaned points decouples rotation+scale from translation.
 */
export function estimateSimilarity(src: Point[], dst: Point[]): Similarity | null {
  const n = src.length;
  if (n < 2) return null;

  let mx = 0, my = 0, mu = 0, mv = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i].x; my += src[i].y;
    mu += dst[i].x; mv += dst[i].y;
  }
  mx /= n; my /= n; mu /= n; mv /= n;

  // a = s·cos θ, b = s·sin θ minimise Σ‖R·p − q‖² over demeaned points
  let numA = 0, numB = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i].x - mx, y = src[i].y - my;
    const u = dst[i].x - mu, v = dst[i].y - mv;
    numA += u * x + v * y;
    numB += v * x - u * y;
    den += x * x + y * y;
  }
  if (den < 1e-10) return null;

  const a = numA / den;
  const b = numB / den;
  return {
    angle: Math.atan2(b, a),
    scale: Math.hypot(a, b),
    tx: mu - (a * mx - b * my),
    ty: mv - (b * mx + a * my),
  };
}

/** Apply a similarity transform to a point. */
export function applySimilarity(t: Similarity, p: Point): Point {
  const a = t.scale * Math.cos(t.angle), b = t.scale * Math.sin(t.angle);
  return { x: a * p.x - b * p.y + t.tx, y: b * p.x + a * p.y + t.ty };
}

export interface RansacResult {
  transform: Similarity;
  /** inliers[i] is true when match i agrees with the final transform */
  inliers: boolean[];
  inlierCount: number;
}

/**
 * RANSAC: repeatedly fit a transform to a random minimal sample (2 matches), count how
 * many matches it explains to within `threshold` pixels, keep the best, then refit with
 * least squares on its inliers.
 */
export function ransacSimilarity(
  src: Point[],
  dst: Point[],
  { iterations = 500, threshold = 3, random = seededRandom(42) }: { iterations?: number; threshold?: number; random?: () => number } = {},
): RansacResult | null {
  const n = src.length;
  if (n < 2) return null;

  const countInliers = (t: Similarity) => {
    const flags = src.map((p, i) => {
      const q = applySimilarity(t, p);
      return Math.hypot(q.x - dst[i].x, q.y - dst[i].y) <= threshold;
    });
    return { flags, count: flags.filter(Boolean).length };
  };

  let best: { flags: boolean[]; count: number } | null = null;
  for (let it = 0; it < iterations; it++) {
    const i = Math.floor(random() * n);
    let j = Math.floor(random() * (n - 1));
    if (j >= i) j++;
    const t = estimateSimilarity([src[i], src[j]], [dst[i], dst[j]]);
    if (!t) continue;
    const scored = countInliers(t);
    if (!best || scored.count > best.count) best = scored;
    if (best.count === n) break;
  }
  if (!best || best.count < 2) return null;

  // Refit on all inliers, then recompute the inlier set once with the refined model.
  const inSrc = src.filter((_, i) => best.flags[i]);
  const inDst = dst.filter((_, i) => best.flags[i]);
  const refined = estimateSimilarity(inSrc, inDst);
  if (!refined) return null;
  const final = countInliers(refined);
  return { transform: refined, inliers: final.flags, inlierCount: final.count };
}

/** Format a transform for display. */
export function formatSimilarity(t: Similarity): string {
  return [
    `θ  = ${(t.angle * 180 / Math.PI).toFixed(1)}°`,
    `s  = ${t.scale.toFixed(3)}`,
    `tx = ${t.tx.toFixed(1)}`,
    `ty = ${t.ty.toFixed(1)}`,
  ].join('\n');
}
