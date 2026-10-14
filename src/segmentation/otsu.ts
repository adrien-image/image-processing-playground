/**
 * Otsu's thresholding.
 *
 * Works on a full 256-bin histogram of 0–255 gray values. The returned threshold `t`
 * splits pixels into background (value ≤ t) and foreground (value > t); `applyThreshold`
 * and the histogram view use exactly the same convention.
 */

/** 256-bin histogram of gray values (rounded to the nearest integer level). */
export function histogram256(pixels: Float64Array): Uint32Array {
  const hist = new Uint32Array(256);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.round(pixels[i]);
    hist[v < 0 ? 0 : v > 255 ? 255 : v]++;
  }
  return hist;
}

/**
 * Otsu's optimal threshold: the `t` maximising between-class variance
 * σ²_B(t) = ω_B(t) · ω_F(t) · (μ_B(t) − μ_F(t))².
 *
 * When several thresholds tie (an empty gap between two peaks), the middle of the
 * tied range is returned, so the cut sits halfway between the peaks.
 */
export function otsuThreshold(hist: Uint32Array): number {
  let total = 0;
  let sum = 0;
  for (let i = 0; i < hist.length; i++) {
    total += hist[i];
    sum += i * hist[i];
  }

  let wB = 0;
  let sumB = 0;
  let maxVariance = -1;
  let bestStart = 0;
  let bestEnd = 0;

  for (let t = 0; t < hist.length; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;

    sumB += t * hist[t];
    const meanB = sumB / wB;
    const meanF = (sum - sumB) / wF;
    const variance = wB * wF * (meanB - meanF) * (meanB - meanF);

    if (variance > maxVariance) {
      maxVariance = variance;
      bestStart = t;
      bestEnd = t;
    } else if (variance === maxVariance) {
      bestEnd = t;
    }
  }

  return Math.round((bestStart + bestEnd) / 2);
}

/** Binary mask: 1 for foreground (value > t), 0 for background (value ≤ t). */
export function applyThreshold(pixels: Float64Array, t: number): Uint8Array {
  const out = new Uint8Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) out[i] = Math.round(pixels[i]) > t ? 1 : 0;
  return out;
}

/**
 * Re-bin a 256-bin histogram for display. Images with only a few distinct levels get
 * fewer, wider bars so the peaks stay visible (8–32 bars). Display only — thresholds are
 * always computed on the full 256 bins.
 */
export function displayHistogram(hist256: Uint32Array): Uint32Array {
  let levels = 0;
  for (let i = 0; i < 256; i++) if (hist256[i] > 0) levels++;
  const bins = Math.min(32, Math.max(8, levels * 4));
  const out = new Uint32Array(bins);
  for (let i = 0; i < 256; i++) out[Math.floor((i * bins) / 256)] += hist256[i];
  return out;
}
