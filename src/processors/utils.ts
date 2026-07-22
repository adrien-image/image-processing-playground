import type { ConvolutionStep } from './types';

/**
 * Apply a 2D kernel via convolution to a single-channel (grayscale) image.
 * Pixels outside the border are treated as 0 (zero-padding).
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
        for (let kx = -halfC; kx <= halfC; kx++) {
          const py = y + ky;
          const px = x + kx;
          const pixel = (py >= 0 && py < height && px >= 0 && px < width)
            ? pixels[py * width + px]
            : 0;
          sum += pixel * kernel[ky + halfR][kx + halfC];
        }
      }
      out[y * width + x] = sum;
    }
  }
  return out;
}

/**
 * Generate step-by-step animation data showing the convolution process
 * for a single pixel path (e.g. a row across the image).
 */
export function generateConvolutionSteps(
  pixels: Float64Array,
  width: number,
  height: number,
  kernel: number[][],
  startCol = 1,
  startRow = 1,
  steps = 10,
): ConvolutionStep[] {
  const kRows = kernel.length;
  const kCols = kernel[0].length;
  const halfR = (kRows - 1) >> 1;
  const halfC = (kCols - 1) >> 1;
  const result: ConvolutionStep[] = [];

  // Walk a snake-like path: right across, then down one row, then left, etc.
  let col = startCol;
  let row = startRow;
  let dir = 1; // 1 = right, -1 = left

  for (let s = 0; s < steps; s++) {
    if (col < 1 || col >= width - 1) {
      row++;
      dir = -dir;
      col = dir === 1 ? 1 : width - 2;
      if (row >= height - 1) break;
    }

    const neighbourhood: number[][] = [];
    const products: number[][] = [];
    let sum = 0;

    for (let ky = -halfR; ky <= halfR; ky++) {
      const nRow: number[] = [];
      const pRow: number[] = [];
      for (let kx = -halfC; kx <= halfC; kx++) {
        const py = row + ky;
        const px = col + kx;
        const pixel = (py >= 0 && py < height && px >= 0 && px < width)
          ? pixels[py * width + px]
          : 0;
        const kVal = kernel[ky + halfR][kx + halfC];
        nRow.push(pixel);
        pRow.push(pixel * kVal);
        sum += pixel * kVal;
      }
      neighbourhood.push(nRow);
      products.push(pRow);
    }

    const output = Math.max(0, Math.min(255, Math.round(sum)));

    result.push({
      col,
      row,
      neighbourhood,
      kernel,
      products,
      sum,
      output,
    });

    col += dir;
  }

  return result;
}

/**
 * Convert RGBA ImageData to grayscale Float64Array (single channel).
 */
export function rgbaToGrayscale(data: Uint8ClampedArray, width: number, height: number): Float64Array {
  const out = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    // Standard luminance weights
    out[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }
  return out;
}

/**
 * Pack a single-channel Float64Array back into RGBA Uint8ClampedArray.
 * Values are clamped to [0, 255].
 */
export function grayscaleToRGBA(pixels: Float64Array, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const v = Math.max(0, Math.min(255, Math.round(pixels[i])));
    out[idx] = v;
    out[idx + 1] = v;
    out[idx + 2] = v;
    out[idx + 3] = 255;
  }
  return out;
}

/** Normalise absolute values in a Float64Array to [0, 255] */
export function normaliseMagnitude(data: Float64Array): Float64Array {
  let max = 0;
  for (let i = 0; i < data.length; i++) { const v = Math.abs(data[i]); if (v > max) max = v; }
  const out = new Float64Array(data.length);
  const s = max > 0 ? 255 / max : 1;
  for (let i = 0; i < data.length; i++) out[i] = Math.abs(data[i]) * s;
  return out;
}
