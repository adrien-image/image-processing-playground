/**
 * Contour lines by marching squares, for drawing a decision function's level sets.
 *
 * `values` holds a function sampled on a res × res grid of cell centres over the extent
 * (row 0 at the top, as in classMap). Each square of four neighbouring samples is crossed
 * by the level set wherever the level lies between two corner values; the crossing point
 * on that edge is found by linear interpolation, and the crossings are joined in pairs.
 */
import type { Extent } from '../shared/plot';

/** Line segments [x₁, y₁, x₂, y₂] in data coordinates where the sampled function equals `level`. */
export function contourSegments(values: ArrayLike<number>, res: number, level: number, extent: Extent): number[][] {
  const px = (col: number) => extent.x[0] + ((col + 0.5) / res) * (extent.x[1] - extent.x[0]);
  const py = (row: number) => extent.y[1] - ((row + 0.5) / res) * (extent.y[1] - extent.y[0]);
  const segments: number[][] = [];
  for (let row = 0; row + 1 < res; row++) {
    for (let col = 0; col + 1 < res; col++) {
      // Corners in order around the square: top-left, top-right, bottom-right, bottom-left.
      const corners: [number, number, number][] = [
        [col, row, values[row * res + col]], [col + 1, row, values[row * res + col + 1]],
        [col + 1, row + 1, values[(row + 1) * res + col + 1]], [col, row + 1, values[(row + 1) * res + col]],
      ];
      const points: [number, number][] = [];
      for (let e = 0; e < 4; e++) {
        const [c0, r0, v0] = corners[e], [c1, r1, v1] = corners[(e + 1) % 4];
        if ((v0 < level) === (v1 < level)) continue;
        const t = (level - v0) / (v1 - v0);
        points.push([px(c0 + t * (c1 - c0)), py(r0 + t * (r1 - r0))]);
      }
      if (points.length === 2) segments.push([...points[0], ...points[1]]);
      else if (points.length === 4) {
        // A saddle: decide which corners connect by the value at the centre.
        const centre = (corners[0][2] + corners[1][2] + corners[2][2] + corners[3][2]) / 4;
        const firstAbove = corners[0][2] >= level;
        if ((centre >= level) === firstAbove) {
          segments.push([...points[0], ...points[1]], [...points[2], ...points[3]]);
        } else {
          segments.push([...points[0], ...points[3]], [...points[1], ...points[2]]);
        }
      }
    }
  }
  return segments;
}

/**
 * Join segments that share end points into polylines [x₀, y₀, x₁, y₁, …], so that a dashed
 * stroke runs continuously along the curve instead of restarting on every tiny segment.
 */
export function joinSegments(segments: number[][]): number[][] {
  const key = (x: number, y: number) => `${x.toFixed(6)},${y.toFixed(6)}`;
  const ends = new Map<string, number[]>(); // end point → indices of the segments touching it
  segments.forEach(([x1, y1, x2, y2], i) => {
    for (const k of [key(x1, y1), key(x2, y2)]) ends.set(k, [...(ends.get(k) ?? []), i]);
  });
  const used = new Uint8Array(segments.length);
  // From (x, y), keep taking an unused segment that touches the current end.
  const walk = (x: number, y: number): number[] => {
    const out: number[] = [];
    for (;;) {
      const next = (ends.get(key(x, y)) ?? []).find(i => !used[i]);
      if (next === undefined) return out;
      used[next] = 1;
      const [x1, y1, x2, y2] = segments[next];
      [x, y] = key(x1, y1) === key(x, y) ? [x2, y2] : [x1, y1];
      out.push(x, y);
    }
  };
  const lines: number[][] = [];
  segments.forEach(([x1, y1, x2, y2], i) => {
    if (used[i]) return;
    used[i] = 1;
    const forward = walk(x2, y2), backward = walk(x1, y1);
    const back: number[] = [];
    for (let k = backward.length - 2; k >= 0; k -= 2) back.push(backward[k], backward[k + 1]);
    lines.push([...back, x1, y1, x2, y2, ...forward]);
  });
  return lines;
}
