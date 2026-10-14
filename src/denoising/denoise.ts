/** Noise models and denoising filters for grayscale images (values 0–255). */

import { gaussianKernel } from '../processors/kernels';
import { convolve, sampleClamped } from '../processors/utils';

export type NoiseType = 'salt-pepper' | 'gaussian';

/** Set a `density` fraction of pixels to 0 or 255 (dead / stuck pixels). */
export function addSaltPepper(pixels: Float64Array, density: number, random: () => number): Float64Array {
  const out = new Float64Array(pixels);
  for (let i = 0; i < out.length; i++) {
    const r = random();
    if (r < density / 2) out[i] = 0;
    else if (r < density) out[i] = 255;
  }
  return out;
}

/** Add zero-mean Gaussian noise of standard deviation `sigma` (sensor noise), clamped to 0–255. */
export function addGaussianNoise(pixels: Float64Array, sigma: number, random: () => number): Float64Array {
  const out = new Float64Array(pixels.length);
  for (let i = 0; i < out.length; i++) {
    // Box–Muller transform
    const n = sigma * Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    out[i] = Math.min(255, Math.max(0, pixels[i] + n));
  }
  return out;
}

/** Median of the (2r+1)² neighbourhood, with replicate padding at the borders. */
export function medianFilter(pixels: Float64Array, w: number, h: number, radius: number): Float64Array {
  const out = new Float64Array(w * h);
  const size = (radius * 2 + 1) ** 2;
  const window = new Float64Array(size);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let k = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) window[k++] = sampleClamped(pixels, w, h, x + dx, y + dy);
      }
      window.sort(); // TypedArray sort is numeric
      out[y * w + x] = window[size >> 1];
    }
  }
  return out;
}

/** σ that OpenCV picks for a k×k Gaussian (k = 2r+1): 0.3·((k−1)/2 − 1) + 0.8. */
export const gaussianSigmaForRadius = (radius: number) => 0.3 * (radius - 1) + 0.8;

/** Gaussian blur over the same (2r+1)² window as the median filter. */
export function gaussianFilter(pixels: Float64Array, w: number, h: number, radius: number): Float64Array {
  return convolve(pixels, w, h, gaussianKernel(radius, gaussianSigmaForRadius(radius)));
}

/** Peak signal-to-noise ratio in dB (higher = closer to the reference). */
export function psnr(reference: Float64Array, test: Float64Array): number {
  let mse = 0;
  for (let i = 0; i < reference.length; i++) {
    const d = reference[i] - test[i];
    mse += d * d;
  }
  mse /= reference.length;
  return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
}
