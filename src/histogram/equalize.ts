/**
 * Histogram equalisation: remap gray levels through the normalised cumulative
 * distribution function (CDF), so the output histogram is spread over 0–255.
 */

import { histogram256 } from '../segmentation/otsu';

export interface EqualizeResult {
  /** Equalised pixels (0–255) */
  output: Float64Array;
  /** mapping[v] = output level for input level v */
  mapping: Float64Array;
}

export function equalize(pixels: Float64Array): EqualizeResult {
  const hist = histogram256(pixels);
  const cdf = new Float64Array(256);
  let running = 0;
  for (let i = 0; i < 256; i++) {
    running += hist[i];
    cdf[i] = running;
  }

  // The lowest occupied level maps to 0 and the highest to 255.
  let cdfMin = 0;
  for (let i = 0; i < 256; i++) {
    if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
  }
  const denom = pixels.length - cdfMin;

  const mapping = new Float64Array(256);
  for (let v = 0; v < 256; v++) {
    // A single-level image has nothing to stretch: leave it unchanged.
    mapping[v] = denom > 0 ? Math.max(0, ((cdf[v] - cdfMin) / denom) * 255) : v;
  }

  const output = new Float64Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.min(255, Math.max(0, Math.round(pixels[i])));
    output[i] = mapping[v];
  }
  return { output, mapping };
}
