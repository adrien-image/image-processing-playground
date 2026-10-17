/**
 * Scale-invariant keypoints: the detector half of SIFT.
 *
 * Blurring an image with Gaussians of growing σ gives a "scale space". Subtracting
 * neighbouring levels gives the difference of Gaussians (DoG), which approximates the
 * scale-normalised Laplacian σ²∇²G: it responds most strongly to a blob whose size
 * matches σ. A keypoint is a DoG value larger (or smaller) than all 26 of its
 * neighbours in space and scale, so it comes with a position *and* a size σ. Zoom the
 * image by 2 and the same blob is found again at 2σ: that is scale invariance.
 *
 * Each octave halves the image and doubles σ, so large blobs cost no more than small
 * ones. Weak responses (low contrast) and responses along edges (where the position is
 * poorly defined) are rejected; the orientation of the strongest local gradient makes
 * the keypoint rotation invariant too.
 */
import { sampleClamped } from '../processors/utils';

export interface Octave {
  width: number;
  height: number;
  /** Pixel size relative to the input: 2^octave. */
  step: number;
  /** Gaussian levels, values 0–1. */
  gaussians: Float64Array[];
  /** dogs[i] = gaussians[i + 1] − gaussians[i] */
  dogs: Float64Array[];
  /** σ of each Gaussian level, in input pixels. */
  sigmas: number[];
}

export interface ScaleSpace {
  octaves: Octave[];
  scalesPerOctave: number;
}

export interface Keypoint {
  /** Position in input pixels. */
  x: number;
  y: number;
  /** Scale in input pixels (the blob's σ). */
  sigma: number;
  /** DoG value: positive for dark blobs on light, negative for light blobs on dark. */
  response: number;
  octave: number;
  /** DoG layer within the octave. */
  layer: number;
  /** Dominant gradient direction in radians. */
  angle: number;
}

/** Separable Gaussian blur with replicate padding. */
export function gaussianBlur(pixels: Float64Array, width: number, height: number, sigma: number): Float64Array {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const kernel = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) sum += kernel[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

  const tmp = new Float64Array(width * height);
  const out = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) s += kernel[k + radius] * sampleClamped(pixels, width, height, x + k, y);
      tmp[y * width + x] = s;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) s += kernel[k + radius] * sampleClamped(tmp, width, height, x, y + k);
      out[y * width + x] = s;
    }
  }
  return out;
}

/** Keep every second pixel in both directions. */
function halve(pixels: Float64Array, width: number, height: number): { pixels: Float64Array; width: number; height: number } {
  const w = Math.floor(width / 2), h = Math.floor(height / 2);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = pixels[2 * y * width + 2 * x];
  return { pixels: out, width: w, height: h };
}

/** The blur a camera image is assumed to have already. */
const INPUT_SIGMA = 0.5;

/**
 * Gaussian and DoG pyramids of a grayscale image (values 0–255). Each octave has
 * `scalesPerOctave + 3` Gaussian levels, σ growing by k = 2^(1/s), so that its DoG
 * layers 1…s have a neighbour above and below.
 */
export function buildScaleSpace(
  gray: Float64Array, width: number, height: number,
  { octaves = 4, scalesPerOctave = 3, sigma0 = 1.6 }: { octaves?: number; scalesPerOctave?: number; sigma0?: number } = {},
): ScaleSpace {
  const s = scalesPerOctave;
  const k = 2 ** (1 / s);
  let base = gaussianBlur(gray.map(v => v / 255), width, height, Math.sqrt(sigma0 ** 2 - INPUT_SIGMA ** 2));
  let w = width, h = height;
  const result: Octave[] = [];
  for (let o = 0; o < octaves && Math.min(w, h) >= 8; o++) {
    const gaussians = [base];
    const local = [sigma0];
    for (let i = 1; i < s + 3; i++) {
      local.push(sigma0 * k ** i);
      // Blurring by a then by b equals blurring by √(a² + b²).
      gaussians.push(gaussianBlur(gaussians[i - 1], w, h, Math.sqrt(local[i] ** 2 - local[i - 1] ** 2)));
    }
    const dogs = gaussians.slice(1).map((g, i) => g.map((v, p) => v - gaussians[i][p]));
    const step = 2 ** o;
    result.push({ width: w, height: h, step, gaussians, dogs, sigmas: local.map(v => v * step) });
    // Level s has σ = 2·σ0: halved, it is the next octave's first level.
    ({ pixels: base, width: w, height: h } = halve(gaussians[s], w, h));
  }
  return { octaves: result, scalesPerOctave: s };
}

/** −½·(f₊ − f₋)/(f₊ + f₋ − 2f₀): where a parabola through three samples peaks, clamped to ±½. */
function parabolaPeak(minus: number, centre: number, plus: number): number {
  const d = plus + minus - 2 * centre;
  return d === 0 ? 0 : Math.max(-0.5, Math.min(0.5, (-0.5 * (plus - minus)) / d));
}

/** Dominant gradient direction around (x, y) in a Gaussian level, from a 36-bin histogram. */
export function dominantOrientation(pixels: Float64Array, width: number, height: number, x: number, y: number, sigma: number): number {
  const bins = new Float64Array(36);
  const ws = 1.5 * sigma;
  const radius = Math.round(3 * ws);
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const px = Math.round(x) + dx, py = Math.round(y) + dy;
      if (px < 1 || py < 1 || px >= width - 1 || py >= height - 1) continue;
      const gx = pixels[py * width + px + 1] - pixels[py * width + px - 1];
      const gy = pixels[(py + 1) * width + px] - pixels[(py - 1) * width + px];
      const weight = Math.exp(-(dx * dx + dy * dy) / (2 * ws * ws)) * Math.hypot(gx, gy);
      const bin = Math.floor(((Math.atan2(gy, gx) + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI) * 36) % 36;
      bins[bin] += weight;
    }
  }
  let best = 0;
  for (let b = 1; b < 36; b++) if (bins[b] > bins[best]) best = b;
  const offset = parabolaPeak(bins[(best + 35) % 36], bins[best], bins[(best + 1) % 36]);
  return ((best + 0.5 + offset) / 36) * 2 * Math.PI;
}

/**
 * Scale-space extrema of the DoG. `contrast` rejects weak responses (DoG units, the
 * image scaled to 0–1); `edgeRatio` rejects points on edges, where one principal
 * curvature is much larger than the other: tr(H)²/det(H) ≥ (r + 1)²/r.
 */
export function detectKeypoints(space: ScaleSpace, contrast = 0.02, edgeRatio = 10): Keypoint[] {
  const keypoints: Keypoint[] = [];
  const s = space.scalesPerOctave;
  const edgeLimit = (edgeRatio + 1) ** 2 / edgeRatio;
  space.octaves.forEach((oct, o) => {
    const { width: w, height: h, dogs } = oct;
    for (let i = 1; i <= s; i++) {
      const [below, here, above] = [dogs[i - 1], dogs[i], dogs[i + 1]];
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const p = y * w + x;
          const v = here[p];
          if (Math.abs(v) < contrast) continue;
          let isMax = true, isMin = true;
          for (const layer of [below, here, above]) {
            for (let dy = -1; dy <= 1 && (isMax || isMin); dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const q = layer[p + dy * w + dx];
                if (layer === here && dx === 0 && dy === 0) continue;
                if (q >= v) isMax = false;
                if (q <= v) isMin = false;
              }
            }
          }
          if (!isMax && !isMin) continue;
          // 2×2 Hessian of the DoG layer: reject edges (and saddles, det ≤ 0).
          const dxx = here[p + 1] + here[p - 1] - 2 * v;
          const dyy = here[p + w] + here[p - w] - 2 * v;
          const dxy = (here[p + w + 1] - here[p + w - 1] - here[p - w + 1] + here[p - w - 1]) / 4;
          const det = dxx * dyy - dxy * dxy;
          if (det <= 0 || (dxx + dyy) ** 2 / det >= edgeLimit) continue;
          // Refine the position and scale with a parabola through each axis.
          const ox = parabolaPeak(here[p - 1], v, here[p + 1]);
          const oy = parabolaPeak(here[p - w], v, here[p + w]);
          const os = parabolaPeak(below[p], v, above[p]);
          const sigma = oct.sigmas[0] * 2 ** ((i + os) / s);
          keypoints.push({
            x: (x + ox) * oct.step, y: (y + oy) * oct.step, sigma, response: v, octave: o, layer: i,
            angle: dominantOrientation(oct.gaussians[i], w, h, x, y, sigma / oct.step),
          });
        }
      }
    }
  });
  return keypoints.sort((a, b) => Math.abs(b.response) - Math.abs(a.response));
}

/**
 * Fraction of the keypoints of view A (those that map inside view B) that are found
 * again in view B at the mapped position and at σ scaled by `zoom`.
 */
export function repeatability(
  a: Keypoint[], b: Keypoint[], toB: (x: number, y: number) => [number, number], zoom: number, width: number, height: number,
): number {
  let visible = 0, found = 0;
  for (const kp of a) {
    const [x, y] = toB(kp.x, kp.y);
    const sigma = kp.sigma * zoom;
    if (x < 2 * sigma || y < 2 * sigma || x > width - 2 * sigma || y > height - 2 * sigma) continue;
    visible++;
    if (b.some(q => Math.hypot(q.x - x, q.y - y) < Math.max(2, 0.6 * sigma) && Math.abs(Math.log2(q.sigma / sigma)) < 0.4)) found++;
  }
  return visible ? found / visible : 0;
}
