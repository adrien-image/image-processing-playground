/**
 * Mathematical morphology on grayscale images with a square structuring element.
 *
 * Erosion takes the minimum over the k×k neighbourhood, dilation the maximum. On a
 * binary (0/255) image this is exactly binary erosion/dilation. A square window is
 * separable, so each pass runs as a 1-D row filter followed by a 1-D column filter
 * (O(k) per pixel instead of O(k²)). Pixels outside the image are ignored.
 */

export type MorphOp = 'erode' | 'dilate' | 'open' | 'close';

function minMax1d(input: Float64Array, width: number, height: number, half: number, horizontal: boolean, takeMin: boolean): Float64Array {
  const out = new Float64Array(input.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let best = takeMin ? Infinity : -Infinity;
      for (let d = -half; d <= half; d++) {
        const px = horizontal ? x + d : x;
        const py = horizontal ? y : y + d;
        if (px < 0 || px >= width || py < 0 || py >= height) continue;
        const v = input[py * width + px];
        if (takeMin ? v < best : v > best) best = v;
      }
      out[y * width + x] = best;
    }
  }
  return out;
}

/** Local minimum over a size×size square: bright regions shrink, dark ones grow. */
export function erode(pixels: Float64Array, width: number, height: number, size: number): Float64Array {
  const half = Math.floor(size / 2);
  return minMax1d(minMax1d(pixels, width, height, half, true, true), width, height, half, false, true);
}

/** Local maximum over a size×size square: bright regions grow, dark ones shrink. */
export function dilate(pixels: Float64Array, width: number, height: number, size: number): Float64Array {
  const half = Math.floor(size / 2);
  return minMax1d(minMax1d(pixels, width, height, half, true, false), width, height, half, false, false);
}

export function morphology(pixels: Float64Array, width: number, height: number, op: MorphOp, size: number): Float64Array {
  switch (op) {
    case 'erode': return erode(pixels, width, height, size);
    case 'dilate': return dilate(pixels, width, height, size);
    // Opening = erode then dilate: removes bright specks smaller than the kernel.
    case 'open': return dilate(erode(pixels, width, height, size), width, height, size);
    // Closing = dilate then erode: fills dark holes smaller than the kernel.
    case 'close': return erode(dilate(pixels, width, height, size), width, height, size);
  }
}

/** Threshold to a binary 0/255 image. */
export function binarize(pixels: Float64Array, threshold: number): Float64Array {
  return pixels.map(v => (v > threshold ? 255 : 0));
}
