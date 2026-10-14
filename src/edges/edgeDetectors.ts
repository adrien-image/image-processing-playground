/**
 * Edge detectors: Sobel, Prewitt, Laplacian and Canny.
 * All convolutions use replicate padding, so the image border is not mistaken for an edge.
 */

import { gaussianKernel, LAPLACIAN_3x3, PREWITT_X, PREWITT_Y, SOBEL_X, SOBEL_Y } from '../processors/kernels';
import { convolve, normaliseMagnitude, sampleClamped } from '../processors/utils';

export type EdgeMethod = 'sobel-gx' | 'sobel-gy' | 'sobel-mag' | 'laplacian' | 'prewitt' | 'canny';

/** √(gx² + gy²) per pixel. */
export function gradientMagnitude(gx: Float64Array, gy: Float64Array): Float64Array {
  const mag = new Float64Array(gx.length);
  for (let i = 0; i < gx.length; i++) mag[i] = Math.hypot(gx[i], gy[i]);
  return mag;
}

/**
 * Non-maximum suppression: keep a pixel only if its gradient magnitude is a local
 * maximum *along the gradient direction*, which thins edges to one pixel.
 *
 * Image coordinates have y pointing down, so a gradient at +45° (gx > 0, gy > 0)
 * points to the lower-right neighbour: compare with (x+1, y+1) and (x−1, y−1).
 */
export function nonMaxSuppression(mag: Float64Array, gx: Float64Array, gy: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  const at = (x: number, y: number) => sampleClamped(mag, w, h, x, y);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const m = mag[idx];
      if (m === 0) continue;
      let a = Math.atan2(gy[idx], gx[idx]) * (180 / Math.PI);
      if (a < 0) a += 180; // direction modulo 180°

      let n1: number, n2: number;
      if (a < 22.5 || a >= 157.5) {        // gradient →: vertical edge
        n1 = at(x - 1, y); n2 = at(x + 1, y);
      } else if (a < 67.5) {                // gradient ↘
        n1 = at(x - 1, y - 1); n2 = at(x + 1, y + 1);
      } else if (a < 112.5) {               // gradient ↓: horizontal edge
        n1 = at(x, y - 1); n2 = at(x, y + 1);
      } else {                              // gradient ↙
        n1 = at(x + 1, y - 1); n2 = at(x - 1, y + 1);
      }
      // ">=" on one side only, so a two-pixel plateau keeps exactly one pixel.
      if (m >= n1 && m > n2) out[idx] = m;
    }
  }
  return out;
}

/**
 * Hysteresis thresholding: pixels ≥ high are edges; pixels ≥ low are edges only if
 * connected (8-neighbourhood, through other such pixels) to a strong one.
 * Implemented as a flood fill from every strong pixel.
 */
export function hysteresis(nms: Float64Array, w: number, h: number, low: number, high: number): Float64Array {
  const edges = new Float64Array(w * h);
  const stack: number[] = [];
  for (let i = 0; i < w * h; i++) {
    if (nms[i] >= high && nms[i] > 0) {
      edges[i] = 255;
      stack.push(i);
    }
  }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w, y = (i - x) / w;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const j = ny * w + nx;
        if (edges[j] === 0 && nms[j] >= low && nms[j] > 0) {
          edges[j] = 255;
          stack.push(j);
        }
      }
    }
  }
  return edges;
}

/**
 * Canny edge detector.
 * 1. Gaussian blur → 2. Sobel gradient → 3. Non-max suppression →
 * 4. Double threshold → 5. Hysteresis edge tracking.
 * `lowPct` / `highPct` are percentages of the strongest suppressed gradient.
 */
export function canny(pixels: Float64Array, w: number, h: number, lowPct: number, highPct: number, sigma = 1): Float64Array {
  return cannyWithGradients(pixels, w, h, lowPct, highPct, sigma).edges;
}

/** Canny, also returning the gradients it computed (used by the Hough circle transform). */
export function cannyWithGradients(pixels: Float64Array, w: number, h: number, lowPct: number, highPct: number, sigma = 1) {
  const blurred = convolve(pixels, w, h, gaussianKernel(2, sigma));
  const gx = convolve(blurred, w, h, SOBEL_X);
  const gy = convolve(blurred, w, h, SOBEL_Y);
  const nms = nonMaxSuppression(gradientMagnitude(gx, gy), gx, gy, w, h);

  let maxMag = 0;
  for (let i = 0; i < nms.length; i++) if (nms[i] > maxMag) maxMag = nms[i];
  const edges = hysteresis(nms, w, h, maxMag * (lowPct / 100), maxMag * (highPct / 100));
  return { edges, gx, gy };
}

export function detectEdges(pixels: Float64Array, w: number, h: number, method: EdgeMethod, cannyLow = 20, cannyHigh = 50): Float64Array {
  switch (method) {
    case 'sobel-gx': return normaliseMagnitude(convolve(pixels, w, h, SOBEL_X));
    case 'sobel-gy': return normaliseMagnitude(convolve(pixels, w, h, SOBEL_Y));
    case 'laplacian': return normaliseMagnitude(convolve(pixels, w, h, LAPLACIAN_3x3));
    case 'prewitt':
      return normaliseMagnitude(gradientMagnitude(convolve(pixels, w, h, PREWITT_X), convolve(pixels, w, h, PREWITT_Y)));
    case 'canny': return canny(pixels, w, h, cannyLow, cannyHigh);
    case 'sobel-mag':
      return normaliseMagnitude(gradientMagnitude(convolve(pixels, w, h, SOBEL_X), convolve(pixels, w, h, SOBEL_Y)));
  }
}
