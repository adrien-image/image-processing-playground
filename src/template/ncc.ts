/**
 * Template matching with normalised cross-correlation (NCC).
 *
 * For every placement of the template, NCC is the correlation between the template
 * and the image window after removing each one's mean and dividing by its standard
 * deviation — so it ranges from −1 to 1 and ignores uniform brightness/contrast changes.
 */

export interface NccMatch {
  x: number;
  y: number;
  score: number;
  /** NCC at every placement: (iw − tw + 1) × (ih − th + 1), row-major */
  scores: Float64Array;
  scoresWidth: number;
  scoresHeight: number;
}

/** Copy a tw×th patch whose top-left corner is (x, y). */
export function extractPatch(gray: Float64Array, width: number, x: number, y: number, tw: number, th: number): Float64Array {
  const patch = new Float64Array(tw * th);
  for (let r = 0; r < th; r++) patch.set(gray.subarray((y + r) * width + x, (y + r) * width + x + tw), r * tw);
  return patch;
}

/** Standard deviation below which a patch is considered flat (no texture to match). */
export const FLAT_STD = 2;

/**
 * Find the best NCC match of `templ` (tw×th) in `image` (iw×ih).
 * Returns null when the template is flat: its standard deviation is ~0, so NCC is undefined.
 *
 * Window means and variances come from integral images (summed-area tables), so they
 * cost O(1) per placement; only the correlation sum is computed per pixel.
 */
export function nccMatch(image: Float64Array, iw: number, ih: number, templ: Float64Array, tw: number, th: number): NccMatch | null {
  const n = tw * th;
  let tMean = 0;
  for (let i = 0; i < n; i++) tMean += templ[i];
  tMean /= n;
  const tz = new Float64Array(n); // zero-mean template
  let tVar = 0;
  for (let i = 0; i < n; i++) {
    tz[i] = templ[i] - tMean;
    tVar += tz[i] * tz[i];
  }
  const tStd = Math.sqrt(tVar / n);
  if (tStd < FLAT_STD) return null;

  // Integral images of I and I² with a zero first row/column.
  const W = iw + 1;
  const sum = new Float64Array(W * (ih + 1));
  const sumSq = new Float64Array(W * (ih + 1));
  for (let y = 0; y < ih; y++) {
    let rowSum = 0, rowSq = 0;
    for (let x = 0; x < iw; x++) {
      const v = image[y * iw + x];
      rowSum += v;
      rowSq += v * v;
      sum[(y + 1) * W + x + 1] = sum[y * W + x + 1] + rowSum;
      sumSq[(y + 1) * W + x + 1] = sumSq[y * W + x + 1] + rowSq;
    }
  }
  const box = (t: Float64Array, x: number, y: number) =>
    t[(y + th) * W + x + tw] - t[y * W + x + tw] - t[(y + th) * W + x] + t[y * W + x];

  const sw = iw - tw + 1, sh = ih - th + 1;
  const scores = new Float64Array(sw * sh);
  let best = { x: 0, y: 0, score: -Infinity };
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const mean = box(sum, x, y) / n;
      const variance = box(sumSq, x, y) / n - mean * mean;
      const iStd = Math.sqrt(Math.max(variance, 0));
      let score = 0;
      if (iStd >= 1e-6) {
        // Σ (I − Ī)(T − T̄) = Σ I·(T − T̄), because Σ (T − T̄) = 0
        let num = 0;
        for (let ty = 0; ty < th; ty++) {
          const row = (y + ty) * iw + x;
          const trow = ty * tw;
          for (let tx = 0; tx < tw; tx++) num += image[row + tx] * tz[trow + tx];
        }
        score = num / (n * tStd * iStd);
      }
      scores[y * sw + x] = score;
      if (score > best.score) best = { x, y, score };
    }
  }
  return { ...best, scores, scoresWidth: sw, scoresHeight: sh };
}
