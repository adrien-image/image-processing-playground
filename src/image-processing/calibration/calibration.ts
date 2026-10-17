/**
 * Lens distortion, and calibrating it from a photo of a checkerboard.
 *
 * A pinhole camera maps straight lines to straight lines; a real lens bends them. The
 * standard radial model moves a point, in normalised coordinates x = (u − cx) / f,
 * along its radius:
 *     x_d = x · (1 + k1·r² + k2·r⁴),   r² = x² + y²
 * k1 < 0 squeezes the corners inward (barrel distortion, wide-angle lenses), k1 > 0
 * pushes them out (pincushion, telephoto). Undistorting inverts the map numerically.
 *
 * Calibration: photograph a board whose corner layout is known. For a guess of k,
 * undistort the detected corners: they are now a plane seen by a pinhole camera, so a
 * homography fits them (a linear least-squares problem). Project the board back through
 * that homography and the lens model, and measure how far the predicted corners land
 * from the detected ones. Gauss–Newton adjusts (k1, k2) to shrink that reprojection
 * error. (Full calibration, Zhang's method, also recovers the focal length and the
 * principal point from several views of the board.)
 */
import { applyHomography, fitHomography, invert, multiply, type Homography, type Point } from '../stitching/homography';

export interface Distortion { k1: number; k2: number }

export interface Camera {
  /** Focal length in pixels. */
  f: number;
  /** Principal point (image centre). */
  cx: number;
  cy: number;
}

export const toNormalised = (cam: Camera, p: Point): Point => ({ x: (p.x - cam.cx) / cam.f, y: (p.y - cam.cy) / cam.f });
export const toPixels = (cam: Camera, p: Point): Point => ({ x: cam.cx + cam.f * p.x, y: cam.cy + cam.f * p.y });

/** Distort a normalised point. */
export function distort({ k1, k2 }: Distortion, p: Point): Point {
  const r2 = p.x * p.x + p.y * p.y;
  const s = 1 + k1 * r2 + k2 * r2 * r2;
  return { x: p.x * s, y: p.y * s };
}

/** Invert `distort` by fixed-point iteration: x = x_d / (1 + k1·r(x)² + k2·r(x)⁴). */
export function undistort(k: Distortion, d: Point, iterations = 30): Point {
  let p = { ...d };
  for (let i = 0; i < iterations; i++) {
    const r2 = p.x * p.x + p.y * p.y;
    const s = 1 + k.k1 * r2 + k.k2 * r2 * r2;
    p = { x: d.x / s, y: d.y / s };
  }
  return p;
}

/** Bilinear RGBA sample; outside the image is black. */
function sample(rgba: Uint8ClampedArray, w: number, h: number, x: number, y: number, out: Uint8ClampedArray, o: number) {
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) {
    out[o] = out[o + 1] = out[o + 2] = 0;
    out[o + 3] = 255;
    return;
  }
  const x0 = Math.min(w - 2, Math.floor(x)), y0 = Math.min(h - 2, Math.floor(y));
  const fx = x - x0, fy = y - y0;
  for (let c = 0; c < 3; c++) {
    const i = (y0 * w + x0) * 4 + c;
    const top = rgba[i] * (1 - fx) + rgba[i + 4] * fx;
    const bottom = rgba[i + w * 4] * (1 - fx) + rgba[i + w * 4 + 4] * fx;
    out[o + c] = top * (1 - fy) + bottom * fy;
  }
  out[o + 3] = 255;
}

/** Warp by inverse mapping: output pixel p takes the input colour at `from(p)`. */
function remap(rgba: Uint8ClampedArray, w: number, h: number, from: (p: Point) => Point): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const s = from({ x, y });
      sample(rgba, w, h, s.x, s.y, out, (y * w + x) * 4);
    }
  }
  return out;
}

/** What the lens does to an ideal (pinhole) image: each distorted pixel looks up its undistorted position. */
export function distortImage(rgba: Uint8ClampedArray, w: number, h: number, cam: Camera, k: Distortion) {
  return remap(rgba, w, h, p => toPixels(cam, undistort(k, toNormalised(cam, p))));
}

/** Correct a photo: each ideal pixel looks up where the lens put it. */
export function undistortImage(rgba: Uint8ClampedArray, w: number, h: number, cam: Camera, k: Distortion) {
  return remap(rgba, w, h, p => toPixels(cam, distort(k, toNormalised(cam, p))));
}

/** Least-squares (k1, k2) from ideal and distorted positions (both normalised). */
export function fitDistortion(ideal: Point[], distorted: Point[]): Distortion {
  // Normal equations of the 2-unknown linear system  [x·r², x·r⁴] · (k1, k2) = x_d − x.
  let a = 0, b = 0, c = 0, r1 = 0, r2 = 0;
  ideal.forEach((p, i) => {
    const q = p.x * p.x + p.y * p.y;
    for (const [u, d] of [[p.x, distorted[i].x], [p.y, distorted[i].y]]) {
      const j1 = u * q, j2 = u * q * q, res = d - u;
      a += j1 * j1; b += j1 * j2; c += j2 * j2;
      r1 += j1 * res; r2 += j2 * res;
    }
  });
  const det = a * c - b * b;
  if (Math.abs(det) < 1e-18) return { k1: 0, k2: 0 };
  return { k1: (c * r1 - b * r2) / det, k2: (a * r2 - b * r1) / det };
}

export interface Calibration {
  k: Distortion;
  /** Board → ideal image homography. */
  H: Homography;
  /** RMS distance (pixels) between the model's predicted corners and the observed ones, per iteration. */
  rms: number[];
}

/** Predicted (distorted) pixel positions of the board corners. */
export function projectCorners(board: Point[], H: Homography, cam: Camera, k: Distortion): Point[] {
  return board.map(p => toPixels(cam, distort(k, toNormalised(cam, applyHomography(H, p)))));
}

/** For a distortion guess, the homography that best fits the undistorted corners. */
function poseFor(board: Point[], observedN: Point[], cam: Camera, k: Distortion): Homography | null {
  return fitHomography(board, observedN.map(p => toPixels(cam, undistort(k, p))));
}

/** Reprojection residuals (x and y, pixels) of a distortion guess, with its best homography. */
function residuals(board: Point[], observed: Point[], observedN: Point[], cam: Camera, k: Distortion) {
  const H = poseFor(board, observedN, cam, k);
  if (!H) return null;
  const predicted = projectCorners(board, H, cam, k);
  return { H, r: predicted.flatMap((p, i) => [p.x - observed[i].x, p.y - observed[i].y]) };
}

const rmsOf = (r: number[]) => Math.sqrt(r.reduce((s, v) => s + v * v, 0) / (r.length / 2));

/**
 * Estimate (k1, k2) and the board pose from detected corners (pixels) by damped
 * Gauss–Newton on k (Levenberg–Marquardt), refitting the homography for every guess.
 * The Jacobian of the residuals with respect to k1 and k2 is taken numerically.
 */
export function calibrate(board: Point[], observed: Point[], cam: Camera, iterations = 20): Calibration {
  const observedN = observed.map(p => toNormalised(cam, p));
  let k: Distortion = { k1: 0, k2: 0 };
  let current = residuals(board, observed, observedN, cam, k);
  if (!current) return { k, H: [1, 0, 0, 0, 1, 0, 0, 0, 1], rms: [] };
  const rms = [rmsOf(current.r)];
  let lambda = 1e-3;
  const eps = 1e-6;
  for (let it = 0; it < iterations; it++) {
    const base = current.r;
    const cols = [{ k1: k.k1 + eps, k2: k.k2 }, { k1: k.k1, k2: k.k2 + eps }]
      .map(kk => residuals(board, observed, observedN, cam, kk)?.r.map((v, i) => (v - base[i]) / eps) ?? null);
    if (!cols[0] || !cols[1]) break;
    const [j1, j2] = cols as number[][];
    // Normal equations (JᵀJ + λ·diag) δ = −Jᵀr for the two unknowns.
    let a = 0, b = 0, c = 0, g1 = 0, g2 = 0;
    for (let i = 0; i < base.length; i++) {
      a += j1[i] * j1[i]; b += j1[i] * j2[i]; c += j2[i] * j2[i];
      g1 += j1[i] * base[i]; g2 += j2[i] * base[i];
    }
    let improved = false;
    for (let attempt = 0; attempt < 10 && !improved; attempt++) {
      const A = a * (1 + lambda), C = c * (1 + lambda), det = A * C - b * b;
      const next = { k1: k.k1 - (C * g1 - b * g2) / det, k2: k.k2 - (A * g2 - b * g1) / det };
      const trial = residuals(board, observed, observedN, cam, next);
      if (trial && rmsOf(trial.r) < rmsOf(base)) {
        k = next; current = trial; lambda = Math.max(1e-7, lambda / 3); improved = true;
      } else {
        lambda *= 10;
      }
    }
    rms.push(rmsOf(current.r));
    if (!improved) break;
  }
  return { k, H: current.H, rms };
}

/** Inner-corner grid of a checkerboard, in board units (one square = 1). */
export function boardCorners(cols: number, rows: number): Point[] {
  const out: Point[] = [];
  for (let j = 1; j <= rows; j++) for (let i = 1; i <= cols; i++) out.push({ x: i, y: j });
  return out;
}

/** Radial displacement (pixels) of a point r pixels from the centre: distorted radius − r. */
export function radialShift(cam: Camera, k: Distortion, r: number): number {
  return distort(k, { x: r / cam.f, y: 0 }).x * cam.f - r;
}

/**
 * An ideal (pinhole) photo of a checkerboard with (cols + 1) × (rows + 1) squares, placed
 * by the board → image homography H; 2×2 supersampling smooths the square edges.
 */
export function renderCheckerboard(w: number, h: number, H: Homography, cols: number, rows: number): Uint8ClampedArray<ArrayBuffer> {
  const Hinv = invert(H);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (const [sx, sy] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
        const b = applyHomography(Hinv, { x: x + sx - 0.5, y: y + sy - 0.5 });
        const onBoard = b.x >= 0 && b.y >= 0 && b.x <= cols + 1 && b.y <= rows + 1;
        v += !onBoard ? 70 : (Math.floor(b.x) + Math.floor(b.y)) % 2 ? 30 : 235;
      }
      const o = (y * w + x) * 4;
      out[o] = out[o + 1] = out[o + 2] = v / 4;
      out[o + 3] = 255;
    }
  }
  return out;
}

/** Board → image homography: board centred at (cx, cy), `scale` pixels per square, rotated, and tilted away along x. */
export function boardPose(cols: number, rows: number, cx: number, cy: number, scale: number, rotateDeg: number, tilt: number): Homography {
  const a = (rotateDeg * Math.PI) / 180, c = Math.cos(a) * scale, s = Math.sin(a) * scale;
  const centreBoard: Homography = [1, 0, -(cols + 1) / 2, 0, 1, -(rows + 1) / 2, 0, 0, 1];
  const rotateScale: Homography = [c, -s, 0, s, c, 0, 0, 0, 1];
  // w = 1 + q·x: points on one side are farther away, so they shrink.
  const perspective: Homography = [1, 0, 0, 0, 1, 0, tilt / (scale * (cols + 1)), 0, 1];
  const toImage: Homography = [1, 0, cx, 0, 1, cy, 0, 0, 1];
  return multiply(toImage, multiply(perspective, multiply(rotateScale, centreBoard)));
}
