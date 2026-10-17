/**
 * Drawing helpers for the lesson's plots (no React): the MSE over a grid of (w, b), its
 * contour lines by marching squares, and a straight boundary clipped to a plot.
 */
import type { Extent } from '../shared/plot';
import { lineMse } from './linear';

export interface Grid {
  res: number;
  /** values[row · res + col], row 0 at the top (largest b). */
  values: Float64Array;
  view: Extent;
}

/** log₁₀ MSE(w, b) on a res × res grid, w along x and b along y. */
export function mseGrid(x: ArrayLike<number>, y: ArrayLike<number>, view: Extent, res: number): Grid {
  const values = new Float64Array(res * res);
  for (let row = 0; row < res; row++) for (let col = 0; col < res; col++) {
    const w = view.x[0] + ((col + 0.5) / res) * (view.x[1] - view.x[0]);
    const b = view.y[1] - ((row + 0.5) / res) * (view.y[1] - view.y[0]);
    values[row * res + col] = Math.log10(lineMse(x, y, w, b) + 1e-9);
  }
  return { res, values, view };
}

/**
 * The contour {value = level} as line segments in grid coordinates (col, row of cell
 * centres), by marching squares: each cell's corners above or below the level pick
 * which edges the line crosses, at linearly interpolated points.
 */
export function contourSegments(values: Float64Array, res: number, level: number): number[] {
  const out: number[] = [];
  const at = (r: number, c: number) => values[r * res + c];
  for (let r = 0; r < res - 1; r++) for (let c = 0; c < res - 1; c++) {
    const v = [at(r, c), at(r, c + 1), at(r + 1, c + 1), at(r + 1, c)];
    const corners: [number, number][] = [[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]];
    const cross: number[] = [];
    for (let e = 0; e < 4; e++) {
      const a = v[e], b = v[(e + 1) % 4];
      if ((a < level) === (b < level)) continue;
      const t = (level - a) / (b - a), [x0, y0] = corners[e], [x1, y1] = corners[(e + 1) % 4];
      cross.push(x0 + t * (x1 - x0), y0 + t * (y1 - y0));
    }
    // Two crossings: one segment. Four (a saddle cell): pair them up in order.
    for (let k = 0; k + 3 < cross.length; k += 4) out.push(cross[k], cross[k + 1], cross[k + 2], cross[k + 3]);
  }
  return out;
}

/** The part of the line a·x + b·y = c inside the extent, or null if it misses it. */
export function clipLine(a: number, b: number, c: number, e: Extent): [number, number, number, number] | null {
  const pts: [number, number][] = [];
  if (Math.abs(b) > 1e-12) for (const x of e.x) {
    const y = (c - a * x) / b;
    if (y >= e.y[0] && y <= e.y[1]) pts.push([x, y]);
  }
  if (Math.abs(a) > 1e-12) for (const y of e.y) {
    const x = (c - b * y) / a;
    if (x >= e.x[0] && x <= e.x[1]) pts.push([x, y]);
  }
  if (pts.length < 2) return null;
  // The two points farthest apart (corners can appear twice).
  let best: [number, number, number, number] = [...pts[0], ...pts[1]], far = -1;
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
    const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
    if (d > far) { far = d; best = [...pts[i], ...pts[j]]; }
  }
  return best;
}
