/**
 * Panorama stitching: a homography from feature matches, then warping and blending.
 *
 * When the camera only rotates (or the scene is flat), two photos are related by a
 * homography: a 3×3 matrix H acting on homogeneous coordinates,
 *     (x', y', w') = H · (x, y, 1),  then divide by w'
 * It has 8 degrees of freedom, so 4 point matches fix it. With more (noisy) matches, the
 * direct linear transform (DLT) finds the least-squares H: each match gives 2 linear
 * equations in the 9 entries, and H is the smallest eigenvector of AᵀA. RANSAC picks
 * random 4-match samples, keeps the H most matches agree with, then refits on those.
 */
import { seededRandom } from '../../lib/random';

export interface Point { x: number; y: number }
export type Homography = number[]; // 9 entries, row-major

export function applyHomography(H: Homography, p: Point): Point {
  const w = H[6] * p.x + H[7] * p.y + H[8];
  return { x: (H[0] * p.x + H[1] * p.y + H[2]) / w, y: (H[3] * p.x + H[4] * p.y + H[5]) / w };
}

export function multiply(A: Homography, B: Homography): Homography {
  const C = new Array<number>(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) C[r * 3 + c] += A[r * 3 + k] * B[k * 3 + c];
  return C;
}

export function invert(H: Homography): Homography {
  const [a, b, c, d, e, f, g, h, i] = H;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map(v => v / det);
}

/** Scale so H[8] = 1, for comparison and display. */
export const normalise = (H: Homography) => H.map(v => v / H[8]);

/**
 * Eigen-decomposition of a symmetric n×n matrix by cyclic Jacobi rotations.
 * Returns eigenvalues and eigenvectors (as columns of `vectors`, row-major n×n).
 */
export function symmetricEigen(M: number[], n: number): { values: number[]; vectors: number[] } {
  const a = [...M];
  const v = Array.from({ length: n * n }, (_, i): number => (i % (n + 1) === 0 ? 1 : 0));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {               // rotate columns p and q
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {               // and rows p and q
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: Array.from({ length: n }, (_, i) => a[i * n + i]), vectors: v };
}

/** Translate and scale points so their centroid is 0 and mean distance √2 (Hartley normalisation). */
function normalisation(points: Point[]): Homography {
  const n = points.length;
  const mx = points.reduce((s, p) => s + p.x, 0) / n, my = points.reduce((s, p) => s + p.y, 0) / n;
  const meanDist = points.reduce((s, p) => s + Math.hypot(p.x - mx, p.y - my), 0) / n || 1;
  const k = Math.SQRT2 / meanDist;
  return [k, 0, -k * mx, 0, k, -k * my, 0, 0, 1];
}

/** Least-squares homography mapping src → dst from ≥ 4 matches (normalised DLT); null if degenerate. */
export function fitHomography(src: Point[], dst: Point[]): Homography | null {
  if (src.length < 4) return null;
  const Ts = normalisation(src), Td = normalisation(dst);
  const A: number[][] = [];
  src.forEach((p0, i) => {
    const p = applyHomography(Ts, p0), q = applyHomography(Td, dst[i]);
    A.push([-p.x, -p.y, -1, 0, 0, 0, q.x * p.x, q.x * p.y, q.x]);
    A.push([0, 0, 0, -p.x, -p.y, -1, q.y * p.x, q.y * p.y, q.y]);
  });
  const AtA = new Array<number>(81).fill(0);
  for (const row of A) for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) AtA[r * 9 + c] += row[r] * row[c];
  const { values, vectors } = symmetricEigen(AtA, 9);
  let k = 0;
  for (let i = 1; i < 9; i++) if (values[i] < values[k]) k = i;
  const h = Array.from({ length: 9 }, (_, r) => vectors[r * 9 + k]);
  const H = multiply(multiply(invert(Td), h), Ts);          // undo the normalisation
  if (!H.every(Number.isFinite) || Math.abs(H[8]) < 1e-12) return null;
  return normalise(H);
}

export interface RansacResult {
  H: Homography;
  inliers: boolean[];
  inlierCount: number;
}

export function ransacHomography(
  src: Point[], dst: Point[],
  { iterations = 400, threshold = 3, random = seededRandom(7) }: { iterations?: number; threshold?: number; random?: () => number } = {},
): RansacResult | null {
  const n = src.length;
  if (n < 4) return null;
  const inliersOf = (H: Homography) => src.map((p, i) => {
    const q = applyHomography(H, p);
    return Math.hypot(q.x - dst[i].x, q.y - dst[i].y) <= threshold;
  });
  let best: boolean[] | null = null, bestCount = 0;
  for (let it = 0; it < iterations; it++) {
    const pick = new Set<number>();
    while (pick.size < 4) pick.add(Math.floor(random() * n));
    const idx = [...pick];
    const H = fitHomography(idx.map(i => src[i]), idx.map(i => dst[i]));
    if (!H) continue;
    const inl = inliersOf(H);
    const count = inl.filter(Boolean).length;
    if (count > bestCount) { best = inl; bestCount = count; }
  }
  if (!best || bestCount < 4) return null;
  const H = fitHomography(src.filter((_, i) => best[i]), dst.filter((_, i) => best[i]));
  if (!H) return null;
  const inliers = inliersOf(H);
  return { H, inliers, inlierCount: inliers.filter(Boolean).length };
}

// ---------------------------------------------------------------- warping and blending

export interface RgbImage {
  /** RGBA bytes */
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** Bilinear RGB sample at (x, y); null outside the image. */
function sampleRgb(img: RgbImage, x: number, y: number): [number, number, number] | null {
  if (x < 0 || y < 0 || x > img.width - 1 || y > img.height - 1) return null;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, img.width - 1), y1 = Math.min(y0 + 1, img.height - 1);
  const fx = x - x0, fy = y - y0;
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const at = (xx: number, yy: number) => img.data[(yy * img.width + xx) * 4 + c];
    out[c] = (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
  }
  return out;
}

export type Blend = 'overwrite' | 'average' | 'feather';

/**
 * Put image A (unchanged) and image B (through H, which maps B's pixels into A's frame)
 * on one canvas big enough for both. Each canvas pixel looks up where it comes from in
 * each image (inverse mapping); where both cover it, `blend` decides:
 *   overwrite  B on top: a hard seam
 *   average    50/50: ghosting if anything is misaligned, and a visible step in brightness
 *   feather    weight each image by the distance to its own border: seams fade out
 */
export function stitch(a: RgbImage, b: RgbImage, H: Homography, blend: Blend) {
  const corners = [[0, 0], [b.width - 1, 0], [0, b.height - 1], [b.width - 1, b.height - 1]].map(([x, y]) => applyHomography(H, { x, y }));
  const minX = Math.floor(Math.min(0, ...corners.map(c => c.x))), maxX = Math.ceil(Math.max(a.width - 1, ...corners.map(c => c.x)));
  const minY = Math.floor(Math.min(0, ...corners.map(c => c.y))), maxY = Math.ceil(Math.max(a.height - 1, ...corners.map(c => c.y)));
  const width = Math.min(2000, maxX - minX + 1), height = Math.min(1500, maxY - minY + 1);
  const Hinv = invert(H);
  const out = new Uint8ClampedArray(width * height * 4);
  const border = (img: RgbImage, x: number, y: number) => Math.max(0, Math.min(x, y, img.width - 1 - x, img.height - 1 - y)) + 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ax = x + minX, ay = y + minY;
      const ca = sampleRgb(a, ax, ay);
      const pb = applyHomography(Hinv, { x: ax, y: ay });
      const cb = sampleRgb(b, pb.x, pb.y);
      let col: [number, number, number] | null = null;
      if (ca && cb) {
        const wa = blend === 'overwrite' ? 0 : blend === 'average' ? 1 : border(a, ax, ay);
        const wb = blend === 'feather' ? border(b, pb.x, pb.y) : 1;
        col = [0, 1, 2].map(c => (ca[c] * wa + cb[c] * wb) / (wa + wb)) as [number, number, number];
      } else col = ca ?? cb;
      const i = (y * width + x) * 4;
      if (col) { out[i] = col[0]; out[i + 1] = col[1]; out[i + 2] = col[2]; out[i + 3] = 255; }
      else { out[i] = out[i + 1] = out[i + 2] = 12; out[i + 3] = 255; }
    }
  }
  return { data: out, width, height, offsetX: minX, offsetY: minY };
}

/** Darken towards the corners, as most lenses do: brightness × (1 − strength · r²), r = 1 at the corners. */
export function vignette(img: RgbImage, strength: number): RgbImage {
  const data = new Uint8ClampedArray(img.data);
  const cx = (img.width - 1) / 2, cy = (img.height - 1) / 2, r2max = cx * cx + cy * cy;
  for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) {
    const f = 1 - strength * (((x - cx) ** 2 + (y - cy) ** 2) / r2max);
    for (let c = 0; c < 3; c++) data[(y * img.width + x) * 4 + c] *= f;
  }
  return { data, width: img.width, height: img.height };
}

/** Render image B of a camera turned by `angleDeg` about the vertical axis, shifted by `shift` px: B(x) = scene(G·x). */
export function cameraHomography(angleDeg: number, shift: number, focal: number, cx: number, cy: number): Homography {
  const t = (angleDeg * Math.PI) / 180;
  const K: Homography = [focal, 0, cx, 0, focal, cy, 0, 0, 1];
  const R: Homography = [Math.cos(t), 0, Math.sin(t), 0, 1, 0, -Math.sin(t), 0, Math.cos(t)];
  const T: Homography = [1, 0, shift, 0, 1, 0, 0, 0, 1];
  return normalise(multiply(T, multiply(K, multiply(R, invert(K)))));
}

/** Warp a scene image through G (output pixel x samples scene at G·x), at the given output size. */
export function renderView(scene: RgbImage, G: Homography, width: number, height: number): RgbImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const p = applyHomography(G, { x, y });
    const c = sampleRgb(scene, p.x, p.y) ?? [0, 0, 0];
    const i = (y * width + x) * 4;
    data[i] = c[0]; data[i + 1] = c[1]; data[i + 2] = c[2]; data[i + 3] = 255;
  }
  return { data, width, height };
}

// ---------------------------------------------------------------- matching

/**
 * Plain patch descriptors: the (2r+1)² gray levels around a corner minus their mean, scaled
 * to 0–1. Unlike the Feature matching lesson's descriptors they are not rotated to a dominant
 * orientation (panorama photos are barely rotated relative to each other) and not divided by
 * their contrast: how bright a window is against its wall is exactly what tells it from its
 * neighbours. Subtracting the mean still forgives an overall brightness change.
 */
export function patchDescriptor(gray: Float64Array, width: number, height: number, p: Point, r = 5): Float64Array | null {
  const x0 = Math.round(p.x), y0 = Math.round(p.y);
  if (x0 < r || y0 < r || x0 >= width - r || y0 >= height - r) return null;
  const d = new Float64Array((2 * r + 1) ** 2);
  let k = 0, mean = 0;
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) mean += d[k++] = gray[(y0 + y) * width + x0 + x] / 255;
  mean /= d.length;
  return d.map(v => v - mean);
}

/** 3×3 binomial blur: makes patches tolerant of the resampling a warp introduces. */
function smooth(gray: Float64Array, width: number, height: number): Float64Array {
  const out = new Float64Array(gray.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let sum = 0, weight = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
      const w = (2 - Math.abs(dx)) * (2 - Math.abs(dy));
      sum += w * gray[yy * width + xx];
      weight += w;
    }
    out[y * width + x] = sum / weight;
  }
  return out;
}

/**
 * Match corners of A to corners of B by descriptor distance, keeping a pair only if it
 * passes Lowe's ratio test and is a mutual best match. Returns matched point pairs.
 */
export function matchPoints(
  grayA: Float64Array, pointsA: Point[], grayB: Float64Array, pointsB: Point[], width: number, height: number, ratio = 0.85,
): { a: Point[]; b: Point[] } {
  const sa = smooth(grayA, width, height), sb = smooth(grayB, width, height);
  const da = pointsA.map(p => patchDescriptor(sa, width, height, p));
  const db = pointsB.map(p => patchDescriptor(sb, width, height, p));
  const dist = (u: Float64Array, v: Float64Array) => {
    let s = 0;
    for (let i = 0; i < u.length; i++) s += (u[i] - v[i]) ** 2;
    return s;
  };
  const D = da.map(u => db.map(v => (u && v ? dist(u, v) : Infinity)));
  const a: Point[] = [], b: Point[] = [];
  D.forEach((row, i) => {
    let best = Infinity, second = Infinity, j = -1;
    row.forEach((d, jj) => {
      if (d < best) { second = best; best = d; j = jj; } else if (d < second) second = d;
    });
    if (j < 0 || !(best < ratio * ratio * second)) return;                      // squared distances
    if (D.some((other, ii) => ii !== i && other[j] < best)) return;            // mutual best
    a.push(pointsA[i]);
    b.push(pointsB[j]);
  });
  return { a, b };
}
