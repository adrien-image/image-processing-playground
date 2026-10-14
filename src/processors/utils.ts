/** Clamp v into [lo, hi]. */
export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Read pixel (x, y) with replicate padding: coordinates outside the image are
 * clamped to the nearest edge pixel.
 */
export function sampleClamped(pixels: Float64Array, width: number, height: number, x: number, y: number): number {
  return pixels[clamp(y, 0, height - 1) * width + clamp(x, 0, width - 1)];
}

/**
 * Slide a kernel over a single-channel image and sum the element-wise products.
 *
 * Like OpenCV's `filter2D` and CNN "convolution" layers, this is technically
 * **cross-correlation**: the kernel is not flipped. For symmetric kernels (Gaussian,
 * Laplacian, sharpen) that is identical to true convolution; for anti-symmetric ones
 * such as Sobel, flipping would only change the sign of the result.
 *
 * Borders use **replicate padding** (clamp-to-edge): pixels outside the image take the
 * value of the nearest edge pixel, so a flat image stays flat and no artificial edge
 * appears around the frame.
 */
export function convolve(
  pixels: Float64Array,
  width: number,
  height: number,
  kernel: number[][],
): Float64Array {
  const kRows = kernel.length;
  const kCols = kernel[0].length;
  const halfR = (kRows - 1) >> 1;
  const halfC = (kCols - 1) >> 1;
  const out = new Float64Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let ky = -halfR; ky <= halfR; ky++) {
        const row = clamp(y + ky, 0, height - 1) * width;
        const kRow = kernel[ky + halfR];
        for (let kx = -halfC; kx <= halfC; kx++) {
          sum += pixels[row + clamp(x + kx, 0, width - 1)] * kRow[kx + halfC];
        }
      }
      out[y * width + x] = sum;
    }
  }
  return out;
}

/**
 * Convert RGBA pixel data to a single grayscale channel (0–255) using the
 * ITU-R BT.601 luma weights.
 */
export function rgbaToGrayscale(data: Uint8ClampedArray, width: number, height: number): Float64Array {
  const out = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    out[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }
  return out;
}

/**
 * Pack a single-channel array back into opaque gray RGBA.
 * Values are rounded and clamped to [0, 255].
 */
export function grayscaleToRGBA(pixels: Float64Array, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const v = clamp(Math.round(pixels[i]), 0, 255);
    out[idx] = v;
    out[idx + 1] = v;
    out[idx + 2] = v;
    out[idx + 3] = 255;
  }
  return out;
}

/** Scale absolute values so the largest becomes 255. */
export function normaliseMagnitude(data: Float64Array): Float64Array {
  let max = 0;
  for (let i = 0; i < data.length; i++) { const v = Math.abs(data[i]); if (v > max) max = v; }
  const out = new Float64Array(data.length);
  const s = max > 0 ? 255 / max : 1;
  for (let i = 0; i < data.length; i++) out[i] = Math.abs(data[i]) * s;
  return out;
}

/** Linearly stretch values so the minimum maps to 0 and the maximum to 255. */
export function normaliseRange(data: Float64Array): Float64Array {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const out = new Float64Array(data.length);
  const range = max - min;
  if (range === 0) return out;
  for (let i = 0; i < data.length; i++) out[i] = ((data[i] - min) / range) * 255;
  return out;
}

/**
 * How raw convolution sums are turned into displayable 0–255 values:
 * - `magnitude`: take |sum| and scale so the strongest response is 255
 *   (edge detectors, whose raw output is signed and unbounded);
 * - `clamp`: keep the sum as-is and clamp to [0, 255]
 *   (smoothing / sharpening kernels whose weights sum to 1).
 */
export type OutputMode = 'magnitude' | 'clamp';

/**
 * Build the function mapping one raw sum to its 0–255 output value. For `magnitude`
 * the scale depends on the strongest response in the whole image, so the full raw
 * result is needed up front.
 */
export function makeOutputMapper(raw: Float64Array, mode: OutputMode): (sum: number) => number {
  if (mode === 'clamp') return (sum) => clamp(Math.round(sum), 0, 255);
  let max = 0;
  for (let i = 0; i < raw.length; i++) { const v = Math.abs(raw[i]); if (v > max) max = v; }
  const scale = max > 0 ? 255 / max : 1;
  return (sum) => clamp(Math.round(Math.abs(sum) * scale), 0, 255);
}
