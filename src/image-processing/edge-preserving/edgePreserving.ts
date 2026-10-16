/**
 * Edge-preserving smoothing: average a pixel only with pixels that look alike.
 *
 * A Gaussian blur weights its neighbours by distance alone, so it averages across
 * edges and blurs them. Two filters add a similarity term to the weight:
 *
 *   bilateral       w(p, q) = exp(−‖p − q‖² / 2σs²) · exp(−(I(q) − I(p))² / 2σr²)
 *   non-local means w(p, q) = exp(−max(d²(P(p), P(q)) − 2σ², 0) / h²)
 *
 * The bilateral filter compares single pixel values: across an edge the values differ
 * by much more than σr, so the weight vanishes and the edge stays sharp. Non-local
 * means compares whole patches around p and q, which is far more robust to noise,
 * and searches a wider window, so a repeated texture averages with its own copies.
 * The −2σ² term discounts the difference that noise alone puts between two copies of
 * the same patch. Images are Float64Array, values 0–255, replicate padding.
 */
import { sampleClamped } from '../processors/utils';

/** Weights of the (2r+1)² window around (x, y), normalised to sum to 1. */
export function bilateralWeights(
  pixels: Float64Array, width: number, height: number, x: number, y: number,
  radius: number, sigmaSpace: number, sigmaRange: number,
): Float64Array {
  const size = 2 * radius + 1;
  const weights = new Float64Array(size * size);
  const centre = pixels[y * width + x];
  let sum = 0;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const diff = sampleClamped(pixels, width, height, x + dx, y + dy) - centre;
      const w = Math.exp(-(dx * dx + dy * dy) / (2 * sigmaSpace * sigmaSpace) - (diff * diff) / (2 * sigmaRange * sigmaRange));
      weights[(dy + radius) * size + dx + radius] = w;
      sum += w;
    }
  }
  for (let i = 0; i < weights.length; i++) weights[i] /= sum;
  return weights;
}

export function bilateralFilter(
  pixels: Float64Array, width: number, height: number, radius: number, sigmaSpace: number, sigmaRange: number,
): Float64Array {
  const out = new Float64Array(width * height);
  const size = 2 * radius + 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const weights = bilateralWeights(pixels, width, height, x, y, radius, sigmaSpace, sigmaRange);
      let s = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          s += weights[(dy + radius) * size + dx + radius] * sampleClamped(pixels, width, height, x + dx, y + dy);
        }
      }
      out[y * width + x] = s;
    }
  }
  return out;
}

export interface NlmOptions {
  /** Patches are (2·patchRadius + 1)² pixels. */
  patchRadius: number;
  /** Candidates come from a (2·searchRadius + 1)² window. */
  searchRadius: number;
  /** Noise standard deviation σ (the expected patch difference 2σ² is discounted). */
  sigma: number;
  /** Filtering strength: larger h averages less similar patches too. */
  h: number;
}

export const DEFAULT_NLM: NlmOptions = { patchRadius: 1, searchRadius: 6, sigma: 20, h: 12 };

/** The image padded by `pad` pixels on each side (replicate), so patches never leave it. */
function padded(pixels: Float64Array, width: number, height: number, pad: number) {
  const pw = width + 2 * pad, ph = height + 2 * pad;
  const out = new Float64Array(pw * ph);
  for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) out[y * pw + x] = sampleClamped(pixels, width, height, x - pad, y - pad);
  return { data: out, width: pw };
}

/** Mean squared difference between the patches around (x1, y1) and (x2, y2) of a padded image. */
function patchDistance(img: Float64Array, pw: number, x1: number, y1: number, x2: number, y2: number, r: number): number {
  let s = 0;
  for (let dy = -r; dy <= r; dy++) {
    const a = (y1 + dy) * pw, b = (y2 + dy) * pw;
    for (let dx = -r; dx <= r; dx++) {
      const d = img[a + x1 + dx] - img[b + x2 + dx];
      s += d * d;
    }
  }
  return s / (2 * r + 1) ** 2;
}

/** Non-local means weights of the search window around (x, y), normalised. */
export function nlmWeights(pixels: Float64Array, width: number, height: number, x: number, y: number, o: NlmOptions): Float64Array {
  const { patchRadius: r, searchRadius: S, sigma, h } = o;
  const pad = r + S;
  const { data, width: pw } = padded(pixels, width, height, pad);
  return windowWeights(data, pw, x + pad, y + pad, r, S, sigma, h);
}

function windowWeights(data: Float64Array, pw: number, px: number, py: number, r: number, S: number, sigma: number, h: number): Float64Array {
  const size = 2 * S + 1;
  const weights = new Float64Array(size * size);
  let sum = 0;
  for (let dy = -S; dy <= S; dy++) {
    for (let dx = -S; dx <= S; dx++) {
      const d2 = patchDistance(data, pw, px, py, px + dx, py + dy, r);
      const w = Math.exp(-Math.max(d2 - 2 * sigma * sigma, 0) / (h * h));
      weights[(dy + S) * size + dx + S] = w;
      sum += w;
    }
  }
  for (let i = 0; i < weights.length; i++) weights[i] /= sum;
  return weights;
}

export function nonLocalMeans(pixels: Float64Array, width: number, height: number, o: NlmOptions): Float64Array {
  const { patchRadius: r, searchRadius: S, sigma, h } = o;
  const pad = r + S;
  const { data, width: pw } = padded(pixels, width, height, pad);
  const size = 2 * S + 1;
  const out = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const px = x + pad, py = y + pad;
      const weights = windowWeights(data, pw, px, py, r, S, sigma, h);
      let s = 0;
      for (let dy = -S; dy <= S; dy++) {
        for (let dx = -S; dx <= S; dx++) s += weights[(dy + S) * size + dx + S] * data[(py + dy) * pw + px + dx];
      }
      out[y * width + x] = s;
    }
  }
  return out;
}
