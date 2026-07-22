/**
 * Otsu's thresholding algorithm.
 * Given a grayscale image (Float64Array, values 0–1), computes the optimal
 * binary threshold by maximising between-class variance.
 */

/** Compute a histogram from grayscale pixel values (0–255) with adapted bin count. */
export function computeHistogram(pixels: Float64Array): Uint32Array {
  // Estimate unique intensity levels: sample the image to determine bin count
  const sample = new Set<number>();
  const step = Math.max(1, Math.floor(pixels.length / 1000));
  for (let i = 0; i < pixels.length; i += step) {
    sample.add(Math.round(pixels[i]));
  }
  const uniqueLevels = sample.size;
  // Use fewer bins for images with few unique values (e.g. checkerboard ~2 levels)
  // so bars are clearly visible. Cap at 32 for wider bars.
  const bins = Math.min(32, Math.max(8, uniqueLevels * 4));

  const hist = new Uint32Array(bins);
  for (let i = 0; i < pixels.length; i++) {
    const bin = Math.min(bins - 1, Math.floor((pixels[i] / 255) * bins));
    hist[bin]++;
  }
  return hist;
}

/**
 * Compute Otsu's optimal threshold from a histogram.
 * Returns the threshold value normalised to 0–1 (fraction of 255).
 * When multiple thresholds give equal variance (gap between peaks),
 * the midpoint of the range is returned.
 */
export function otsuThreshold(hist: Uint32Array, totalPixels: number): number {
  const bins = hist.length;
  let sum = 0;
  for (let i = 0; i < bins; i++) sum += i * hist[i];

  let sumB = 0;
  let wB = 0;
  let maxVariance = 0;
  let bestTStart = 0;
  let bestTEnd = 0;

  for (let t = 0; t < bins; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = totalPixels - wB;
    if (wF === 0) break;

    sumB += t * hist[t];
    const meanB = sumB / wB;
    const meanF = (sum - sumB) / wF;

    const variance = wB * wF * (meanB - meanF) * (meanB - meanF);

    if (variance > maxVariance) {
      maxVariance = variance;
      bestTStart = t;
      bestTEnd = t;
    } else if (variance === maxVariance && variance > 0) {
      // Extend the range of equal-variance thresholds
      bestTEnd = t;
    }
  }

  // Use the midpoint of the equal-variance range
  const threshold = Math.round((bestTStart + bestTEnd) / 2);
  return threshold / bins;
}

/** Apply a threshold to grayscale pixels (0–255). Returns binary mask (0 or 1). */
export function applyThreshold(pixels: Float64Array, threshold: number): Float64Array {
  const out = new Float64Array(pixels.length);
  const t = threshold * 255;
  for (let i = 0; i < pixels.length; i++) {
    out[i] = pixels[i] >= t ? 1 : 0;
  }
  return out;
}

