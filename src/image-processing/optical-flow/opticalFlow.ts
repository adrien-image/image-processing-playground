/**
 * Optical flow: how far did each part of the image move between two frames?
 *
 * Lucas–Kanade assumes brightness is preserved, I₂(p + d) = I₁(p), and that the motion d
 * is the same over a small window. Linearising, every pixel in the window gives one
 * equation Ix·u + Iy·v = −It, and least squares over the window gives the 2×2 system
 *
 *     [ΣIx²   ΣIxIy] [u]     [ΣIxIt]
 *     [ΣIxIy  ΣIy² ] [v] = − [ΣIyIt]
 *
 * It only works when that matrix is well conditioned: along a straight edge only the
 * motion across the edge can be measured (the aperture problem), which shows up as a
 * small minimum eigenvalue. And the linearisation only holds for motions of a pixel or
 * so; a pyramid handles larger ones: estimate on a half-size image, double, refine.
 */

export type Motion = 'translate' | 'rotate' | 'zoom';

export interface Frame {
  pixels: Float64Array;
  width: number;
  height: number;
}

// ---------------------------------------------------------------- synthetic frames

/**
 * A smooth texture with detail at several scales, as natural images have (0–255).
 * The coarse components are what lets a pyramid track large motions.
 */
export function texture(x: number, y: number): number {
  return 128
    + 30 * Math.sin(x * 0.045 + 1) * Math.cos(y * 0.05 - 0.5)
    + 22 * Math.cos((x - 2 * y) * 0.03)
    + 25 * Math.sin(x * 0.23 + Math.sin(y * 0.07) * 2)
    + 20 * Math.cos(y * 0.19 - x * 0.05)
    + 18 * Math.sin((x + y) * 0.11) * Math.cos((x - y) * 0.13);
}

/** Vertical stripes: every point on a stripe looks the same, so vertical motion is invisible. */
export function stripes(x: number): number {
  return 128 + 90 * Math.sin(x * 0.35);
}

/** The motion as a map from frame-1 positions to frame-2 positions, about the image centre. */
export function motionMap(kind: Motion, amount: number, cx: number, cy: number) {
  return (x: number, y: number): [number, number] => {
    if (kind === 'translate') return [x + amount, y + amount * 0.5];
    const dx = x - cx, dy = y - cy;
    if (kind === 'zoom') return [cx + dx * (1 + amount / 100), cy + dy * (1 + amount / 100)];
    const a = (amount * Math.PI) / 180;
    return [cx + dx * Math.cos(a) - dy * Math.sin(a), cy + dx * Math.sin(a) + dy * Math.cos(a)];
  };
}

/** Frame 1 samples the pattern; frame 2 samples it through the inverse motion, so content moves by the motion. */
export function makeFrames(kind: Motion, amount: number, width: number, height: number, pattern: 'texture' | 'stripes') {
  const f = pattern === 'texture' ? texture : (x: number) => stripes(x);
  const cx = width / 2, cy = height / 2;
  const forward = motionMap(kind, amount, cx, cy);
  const inverse = motionMap(kind, kind === 'zoom' ? (100 / (1 + amount / 100) - 100) : -amount, cx, cy);
  const a = new Float64Array(width * height), b = new Float64Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    a[y * width + x] = f(x, y);
    const [sx, sy] = inverse(x, y);
    b[y * width + x] = f(sx, sy);
  }
  /** True displacement of the content at frame-1 pixel (x, y). */
  const truth = (x: number, y: number): [number, number] => {
    const [x2, y2] = forward(x, y);
    return [x2 - x, y2 - y];
  };
  return { frame1: { pixels: a, width, height }, frame2: { pixels: b, width, height }, truth };
}

// ---------------------------------------------------------------- Lucas–Kanade

function sample(f: Frame, x: number, y: number): number {
  const cx = Math.min(Math.max(x, 0), f.width - 1), cy = Math.min(Math.max(y, 0), f.height - 1);
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, f.width - 1), y1 = Math.min(y0 + 1, f.height - 1);
  const fx = cx - x0, fy = cy - y0, p = f.pixels, w = f.width;
  return (p[y0 * w + x0] * (1 - fx) + p[y0 * w + x1] * fx) * (1 - fy) + (p[y1 * w + x0] * (1 - fx) + p[y1 * w + x1] * fx) * fy;
}

const BINOMIAL = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];

/** Half-size frame: blur with the binomial filter [1 4 6 4 1]/16, keep every second pixel (see the Pyramids lesson). */
export function downsample(f: Frame): Frame {
  const { width: w, height: h, pixels } = f;
  const at = (x: number, y: number) => pixels[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  const nw = Math.ceil(w / 2), nh = Math.ceil(h / 2);
  const out = new Float64Array(nw * nh);
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
    let s = 0;
    for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) s += BINOMIAL[j + 2] * BINOMIAL[i + 2] * at(2 * x + i, 2 * y + j);
    out[y * nw + x] = s;
  }
  return { pixels: out, width: nw, height: nh };
}

export interface FlowVector {
  x: number;
  y: number;
  u: number;
  v: number;
  /** Smaller eigenvalue of the structure tensor, per pixel of the window: ≈ 0 means the motion is ambiguous. */
  minEigen: number;
}

/**
 * Lucas–Kanade at one point: `iterations` Gauss–Newton steps, each warping frame 2 by the
 * current estimate (u, v) and solving the 2×2 system for the remaining motion.
 */
export function lucasKanadePoint(f1: Frame, f2: Frame, x: number, y: number, radius: number, u0 = 0, v0 = 0, iterations = 5): FlowVector {
  let u = u0, v = v0;
  let minEigen = 0;
  for (let it = 0; it < iterations; it++) {
    let sxx = 0, sxy = 0, syy = 0, sxt = 0, syt = 0;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = x + dx, py = y + dy;
        // Gradients of the average of both frames (more symmetric, more accurate than frame 1 alone).
        const ix = (sample(f1, px + 1, py) - sample(f1, px - 1, py) + sample(f2, px + u + 1, py + v) - sample(f2, px + u - 1, py + v)) / 4;
        const iy = (sample(f1, px, py + 1) - sample(f1, px, py - 1) + sample(f2, px + u, py + v + 1) - sample(f2, px + u, py + v - 1)) / 4;
        const it2 = sample(f2, px + u, py + v) - sample(f1, px, py);
        sxx += ix * ix; sxy += ix * iy; syy += iy * iy;
        sxt += ix * it2; syt += iy * it2;
      }
    }
    const det = sxx * syy - sxy * sxy;
    const n = (2 * radius + 1) ** 2;
    const tr = sxx + syy;
    const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    const big = tr / 2 + disc, small = tr / 2 - disc;
    minEigen = small / n;
    if (big < 1e-9) break; // a flat window: nothing to track
    let du: number, dv: number;
    if (small > 1e-6 * big) {
      du = (-syy * sxt + sxy * syt) / det;
      dv = (sxy * sxt - sxx * syt) / det;
    } else {
      // Aperture problem: only the motion along the gradient is measurable. Take the
      // smallest solution: project onto the main eigenvector e, d = −(e·b / λ) e.
      let ex = sxy, ey = big - sxx;
      if (Math.hypot(ex, ey) < 1e-12) { ex = big - syy; ey = sxy; }
      const norm = Math.hypot(ex, ey) || 1;
      ex /= norm; ey /= norm;
      const k = -(ex * sxt + ey * syt) / big;
      du = k * ex;
      dv = k * ey;
    }
    u += du;
    v += dv;
    if (Math.hypot(du, dv) < 0.01) break;
  }
  return { x, y, u, v, minEigen };
}

/**
 * Pyramidal Lucas–Kanade at the given points: start at the coarsest of `levels`, where all
 * motions are `2^(levels−1)` times smaller, then double the estimate and refine at each finer level.
 */
export function lucasKanade(f1: Frame, f2: Frame, points: { x: number; y: number }[], radius: number, levels: number): FlowVector[] {
  const p1 = [f1], p2 = [f2];
  for (let l = 1; l < levels; l++) { p1.push(downsample(p1[l - 1])); p2.push(downsample(p2[l - 1])); }
  return points.map(({ x, y }) => {
    let u = 0, v = 0;
    let result: FlowVector = { x, y, u: 0, v: 0, minEigen: 0 };
    for (let l = levels - 1; l >= 0; l--) {
      const s = 2 ** l;
      result = lucasKanadePoint(p1[l], p2[l], x / s, y / s, radius, u, v);
      u = result.u * (l > 0 ? 2 : 1);
      v = result.v * (l > 0 ? 2 : 1);
    }
    return { x, y, u: result.u, v: result.v, minEigen: result.minEigen };
  });
}

/** Points on a regular grid, `margin` px away from the borders. */
export function gridPoints(width: number, height: number, step: number, margin: number) {
  const pts: { x: number; y: number }[] = [];
  for (let y = margin; y < height - margin; y += step) for (let x = margin; x < width - margin; x += step) pts.push({ x, y });
  return pts;
}
