/**
 * Similarity transform estimation via least-squares (rotation + translation).
 * Pure TypeScript — no external dependencies, no RANSAC.
 */

export interface SimTransformResult {
  /** Rotation angle in radians */
  angle: number;
  /** Translation in pixels */
  tx: number;
  ty: number;
  /** Number of matches used */
  numMatches: number;
}

interface Point { x: number; y: number; }

/**
 * Estimate similarity transform (rotation + translation) from point correspondences
 * using least-squares. Solves for θ, tx, ty where:
 *   x' = cos(θ)*x - sin(θ)*y + tx
 *   y' = sin(θ)*x + cos(θ)*y + ty
 *
 * The solution uses centroid-demeaned points to decouple rotation from translation.
 */
export function estimateTransform(
  srcPoints: Point[],
  dstPoints: Point[],
): SimTransformResult {
  const numMatches = srcPoints.length;
  if (numMatches < 2) {
    return { angle: 0, tx: 0, ty: 0, numMatches };
  }

  // Compute centroids
  let mx = 0, my = 0, mu = 0, mv = 0;
  for (let i = 0; i < numMatches; i++) {
    mx += srcPoints[i].x; my += srcPoints[i].y;
    mu += dstPoints[i].x; mv += dstPoints[i].y;
  }
  mx /= numMatches; my /= numMatches;
  mu /= numMatches; mv /= numMatches;

  // Solve for c = cos(θ) and s = sin(θ) using demeaned points
  let num = 0, den = 0, numS = 0;
  for (let i = 0; i < numMatches; i++) {
    const x = srcPoints[i].x - mx, y = srcPoints[i].y - my;
    const u = dstPoints[i].x - mu, v = dstPoints[i].y - mv;
    num += u * x + v * y;     // numerator for c
    numS += v * x - u * y;    // numerator for s
    den += x * x + y * y;
  }

  let angle = 0, tx = 0, ty = 0;

  if (Math.abs(den) > 1e-10) {
    const c = num / den;
    const s = numS / den;
    angle = Math.atan2(s, c);
    tx = mu - (c * mx - s * my);
    ty = mv - (s * mx + c * my);
  }

  return { angle, tx, ty, numMatches };
}

/**
 * Format the transform result as a display string.
 */
export function formatTransform(r: SimTransformResult): string {
  const deg = (r.angle * 180 / Math.PI).toFixed(1);
  return `θ = ${deg}°\ntx = ${r.tx.toFixed(1)}\nty = ${r.ty.toFixed(1)}`;
}
