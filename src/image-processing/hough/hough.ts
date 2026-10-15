/**
 * Hough transforms for lines and circles, run on a thin (Canny) edge map.
 */

/** Hot colormap: 0 = black → red → yellow → 1 = white. Returns 0–255 RGB. */
export function hotColor(t: number): [number, number, number] {
  const c = Math.min(1, Math.max(0, t));
  return [
    Math.min(1, c * 3) * 255,
    Math.min(1, Math.max(0, c * 3 - 1)) * 255,
    Math.min(1, Math.max(0, c * 3 - 2)) * 255,
  ];
}

// ---------------------------------------------------------------- lines

export interface LineSegment {
  /** Normal form: x·cos θ + y·sin θ = ρ */
  rho: number;
  theta: number;
  votes: number;
  /** Accumulator cell of the peak */
  thetaIndex: number;
  rhoIndex: number;
  /** Endpoints of the longest supported stretch of the line */
  x1: number; y1: number; x2: number; y2: number;
  length: number;
}

export interface HoughLinesResult {
  lines: LineSegment[];
  /** Votes, indexed [thetaIndex * numRho + rhoIndex] */
  accumulator: Uint32Array;
  numTheta: number;
  numRho: number;
}

export interface LineOptions {
  /** Minimum votes (edge pixels on the line) for a peak */
  threshold: number;
  /** Minimum length of the detected segment, in pixels */
  minLength: number;
  /** Largest gap (pixels) allowed inside one segment */
  maxGap?: number;
  maxLines?: number;
  /**
   * Lines closer than this (in degrees and in pixels of ρ) to a stronger line are treated
   * as duplicates — e.g. the two edges of one thick stroke.
   */
  mergeAngle?: number;
  mergeDistance?: number;
}

const NUM_THETA = 180; // 1° steps over [0°, 180°)

/**
 * Each edge pixel votes for every line (ρ, θ) through it. Peaks in the accumulator
 * are lines; each peak is then traced along the image to find the actual segment.
 */
export function houghLines(edges: Float64Array, w: number, h: number, opts: LineOptions): HoughLinesResult {
  const { threshold, minLength, maxGap = 4, maxLines = 20, mergeAngle = 4, mergeDistance = 6 } = opts;
  const maxRho = Math.ceil(Math.hypot(w, h));
  const numRho = 2 * maxRho + 1; // ρ ∈ [−maxRho, maxRho] in 1-px bins
  const cos = new Float64Array(NUM_THETA), sin = new Float64Array(NUM_THETA);
  for (let t = 0; t < NUM_THETA; t++) {
    cos[t] = Math.cos((t * Math.PI) / NUM_THETA);
    sin[t] = Math.sin((t * Math.PI) / NUM_THETA);
  }

  const acc = new Uint32Array(NUM_THETA * numRho);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (edges[y * w + x] < 128) continue;
      for (let t = 0; t < NUM_THETA; t++) {
        acc[t * numRho + Math.round(x * cos[t] + y * sin[t]) + maxRho]++;
      }
    }
  }

  // Local maxima. θ wraps around: (θ = 180°, ρ) is the same line as (0°, −ρ).
  const vote = (t: number, r: number) => {
    if (t < 0) { t += NUM_THETA; r = numRho - 1 - r; }
    else if (t >= NUM_THETA) { t -= NUM_THETA; r = numRho - 1 - r; }
    return r < 0 || r >= numRho ? 0 : acc[t * numRho + r];
  };
  const peaks: { t: number; r: number; votes: number }[] = [];
  for (let t = 0; t < NUM_THETA; t++) {
    for (let r = 0; r < numRho; r++) {
      const v = acc[t * numRho + r];
      if (v < threshold) continue;
      let isMax = true;
      for (let dt = -2; dt <= 2 && isMax; dt++) {
        for (let dr = -4; dr <= 4 && isMax; dr++) {
          if (dt === 0 && dr === 0) continue;
          const n = vote(t + dt, r + dr);
          // Ties: the earlier cell wins, so a plateau yields one peak.
          if (n > v || (n === v && (dt < 0 || (dt === 0 && dr < 0)))) isMax = false;
        }
      }
      if (isMax) peaks.push({ t, r, votes: v });
    }
  }
  peaks.sort((a, b) => b.votes - a.votes);

  const isDuplicate = (t: number, rho: number) => lines.some(l => {
    let dt = Math.abs(t - l.thetaIndex), dr = Math.abs(rho - l.rho);
    if (dt > NUM_THETA / 2) { dt = NUM_THETA - dt; dr = Math.abs(rho + l.rho); } // θ wraps, ρ flips
    return dt <= (mergeAngle * NUM_THETA) / 180 && dr <= mergeDistance;
  });

  const lines: LineSegment[] = [];
  for (const p of peaks) {
    if (isDuplicate(p.t, p.r - maxRho)) continue;
    const seg = traceSegment(edges, w, h, cos[p.t], sin[p.t], p.r - maxRho, maxGap);
    // A peak a few degrees off a strong line can still cross that line's (thick) edge
    // band for a while; skip segments that mostly run along an accepted one.
    if (seg && seg.length >= minLength && !mostlyAlong(seg, lines, mergeDistance / 2)) {
      lines.push({ rho: p.r - maxRho, theta: (p.t * Math.PI) / NUM_THETA, votes: p.votes, thetaIndex: p.t, rhoIndex: p.r, ...seg });
      if (lines.length >= maxLines) break;
    }
  }
  return { lines, accumulator: acc, numTheta: NUM_THETA, numRho };
}

type Segment = { x1: number; y1: number; x2: number; y2: number; length: number };

function distanceToSegment(px: number, py: number, s: Segment): number {
  const dx = s.x2 - s.x1, dy = s.y2 - s.y1;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, ((px - s.x1) * dx + (py - s.y1) * dy) / len2)) : 0;
  return Math.hypot(px - (s.x1 + t * dx), py - (s.y1 + t * dy));
}

/** True when at least half of `seg` lies within `tolerance` px of one of `accepted`. */
function mostlyAlong(seg: Segment, accepted: Segment[], tolerance: number): boolean {
  const steps = Math.max(1, Math.ceil(seg.length / 2));
  let near = 0;
  for (let i = 0; i <= steps; i++) {
    const x = seg.x1 + ((seg.x2 - seg.x1) * i) / steps, y = seg.y1 + ((seg.y2 - seg.y1) * i) / steps;
    if (accepted.some(a => distanceToSegment(x, y, a) <= tolerance)) near++;
  }
  return near / (steps + 1) >= 0.5;
}

/** Walk along a line and return its longest run of edge support (gaps ≤ maxGap). */
function traceSegment(edges: Float64Array, w: number, h: number, c: number, s: number, rho: number, maxGap: number) {
  const reach = Math.hypot(w, h);
  const px = rho * c, py = rho * s; // closest point to the origin
  const dx = -s, dy = c;            // direction along the line
  const hit = (x: number, y: number) => {
    // Accept edge pixels within ±1 px across the line (discretisation slack).
    for (let k = -1; k <= 1; k++) {
      const ix = Math.round(x + k * c), iy = Math.round(y + k * s);
      if (ix >= 0 && ix < w && iy >= 0 && iy < h && edges[iy * w + ix] >= 128) return true;
    }
    return false;
  };

  let best: { a: number; b: number } | null = null;
  let start = NaN, last = NaN;
  for (let t = -reach; t <= reach; t += 0.5) {
    if (!hit(px + t * dx, py + t * dy)) continue;
    if (Number.isNaN(start) || t - last > maxGap) start = t;
    last = t;
    if (!best || last - start > best.b - best.a) best = { a: start, b: last };
  }
  if (!best) return null;
  return {
    x1: px + best.a * dx, y1: py + best.a * dy,
    x2: px + best.b * dx, y2: py + best.b * dy,
    length: best.b - best.a,
  };
}

// ---------------------------------------------------------------- circles

export interface Circle {
  x: number;
  y: number;
  r: number;
  /** Votes at (and, partly weighted, next to) the centre */
  votes: number;
  /** votes / circumference — roughly the fraction of the circle that is supported */
  score: number;
}

export interface HoughCirclesResult {
  circles: Circle[];
  /** Best score over all radii for each centre, 0–1-ish, for display */
  centreMap: Float64Array;
}

export interface CircleOptions {
  minR: number;
  maxR: number;
  /** Minimum score (fraction of the circumference supported) */
  threshold: number;
  maxCircles?: number;
}

/**
 * Gradient-directed circle Hough: an edge pixel on a circle has its gradient along
 * the radius, so it only votes for the two centres at distance r along ±gradient
 * (instead of a whole ring of candidate centres). Peaks are found in (x, y, r) space
 * so the same circle is not reported at several neighbouring radii.
 */
export function houghCircles(
  edges: Float64Array, gx: Float64Array, gy: Float64Array, w: number, h: number, opts: CircleOptions,
): HoughCirclesResult {
  const { minR, maxR, threshold, maxCircles = 10 } = opts;
  const numR = maxR - minR + 1;
  const plane = w * h;
  const acc = new Uint16Array(numR * plane);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (edges[i] < 128) continue;
      const mag = Math.hypot(gx[i], gy[i]);
      if (mag === 0) continue;
      const ux = gx[i] / mag, uy = gy[i] / mag;
      for (let ri = 0; ri < numR; ri++) {
        const r = minR + ri;
        for (const sign of [1, -1]) {
          const cx = Math.round(x + sign * r * ux), cy = Math.round(y + sign * r * uy);
          if (cx >= 0 && cx < w && cy >= 0 && cy < h) acc[ri * plane + cy * w + cx]++;
        }
      }
    }
  }

  // Rounding scatters a circle's votes over neighbouring cells, so also count votes
  // one cell away — at half weight (quarter on diagonals). Votes cast with the wrong
  // radius land on a ring around the centre and so count much less than the true radius.
  const smoothed = new Float64Array(numR * plane);
  for (let ri = 0; ri < numR; ri++) {
    const base = ri * plane;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx >= 0 && nx < w) sum += acc[base + ny * w + nx] / ((dx ? 2 : 1) * (dy ? 2 : 1));
          }
        }
        smoothed[base + y * w + x] = sum;
      }
    }
  }
  const score = (ri: number, idx: number) => smoothed[ri * plane + idx] / (2 * Math.PI * (minR + ri));
  const centreMap = new Float64Array(plane);
  const candidates: Circle[] = [];
  const R = 3; // suppression radius in x, y and r
  for (let ri = 0; ri < numR; ri++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const s = score(ri, idx);
        if (s > centreMap[idx]) centreMap[idx] = s;
        if (s < threshold) continue;
        let isMax = true;
        for (let dr = -R; dr <= R && isMax; dr++) {
          const nr = ri + dr;
          if (nr < 0 || nr >= numR) continue;
          for (let dy = -R; dy <= R && isMax; dy++) {
            for (let dx = -R; dx <= R && isMax; dx++) {
              if (!dr && !dx && !dy) continue;
              const nx = x + dx, ny = y + dy;
              if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
              const n = score(nr, ny * w + nx);
              if (n > s || (n === s && (dr < 0 || (dr === 0 && (dy < 0 || (dy === 0 && dx < 0)))))) isMax = false;
            }
          }
        }
        if (isMax) candidates.push({ x, y, r: minR + ri, votes: smoothed[ri * plane + idx], score: s });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  // A stroked circle has two edges (inside and outside the stroke): drop weaker
  // detections that are essentially the same circle as a stronger one.
  const circles: Circle[] = [];
  for (const c of candidates) {
    if (circles.some(k => Math.hypot(k.x - c.x, k.y - c.y) <= 4 && Math.abs(k.r - c.r) <= 6)) continue;
    circles.push(c);
    if (circles.length >= maxCircles) break;
  }
  return { circles, centreMap };
}
