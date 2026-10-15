/**
 * A `Plane` is one channel of an image: `width × height` numbers (0–255 for pixel
 * values, but any real number is allowed — Laplacian bands are signed) stored row by
 * row, so pixel (x, y) lives at index `y * width + x`.
 *
 * Colour images are handled as three planes (R, G, B) that are processed
 * independently and recombined at the end.
 */
export interface Plane {
  data: Float64Array;
  width: number;
  height: number;
}

/** A new plane filled with `value`. */
export function createPlane(width: number, height: number, value = 0): Plane {
  return { data: new Float64Array(width * height).fill(value), width, height };
}

/** Split interleaved RGBA bytes (e.g. `ImageData.data`) into R, G and B planes; alpha is dropped. */
export function splitRGB(rgba: ArrayLike<number>, width: number, height: number): Plane[] {
  const planes = [0, 1, 2].map(() => createPlane(width, height));
  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) planes[c].data[i] = rgba[i * 4 + c];
  }
  return planes;
}

/**
 * Pack planes back into opaque RGBA bytes: three planes give colour, one plane gives gray.
 * Each value is mapped through `offset + gain · v`, rounded and clamped to 0–255, so a
 * signed band can be shown with 0 as mid-gray (`offset = 128`).
 */
export function mergeRGB(planes: readonly Plane[], offset = 0, gain = 1): Uint8ClampedArray<ArrayBuffer> {
  const { width, height } = planes[0];
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) {
      const plane = planes[planes.length === 1 ? 0 : c];
      out[i * 4 + c] = Math.round(offset + gain * plane.data[i]); // Uint8ClampedArray clamps to 0–255
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Pixel-wise a + b (same size). */
export function addPlanes(a: Plane, b: Plane): Plane {
  const out = createPlane(a.width, a.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  return out;
}

/** Pixel-wise a − b (same size). */
export function subtractPlanes(a: Plane, b: Plane): Plane {
  const out = createPlane(a.width, a.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = a.data[i] - b.data[i];
  return out;
}

/** Largest |a − b| over all pixels (0 when the planes are identical). */
export function maxAbsDifference(a: Plane, b: Plane): number {
  let max = 0;
  for (let i = 0; i < a.data.length; i++) max = Math.max(max, Math.abs(a.data[i] - b.data[i]));
  return max;
}

/**
 * Mirror an index into 0 … n−1 without repeating the edge pixel ("reflect-101"):
 * …, 2, 1, | 0, 1, 2, …, n−1, | n−2, n−3, …
 */
export function mirrorIndex(i: number, n: number): number {
  if (i >= 0 && i < n) return i;
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  const j = ((i % period) + period) % period;
  return j < n ? j : period - j;
}

/**
 * Filter every row of a w × h buffer with a 1-D kernel and write the result
 * **transposed** (as an h × w buffer: row x of the output is column x of the input).
 * Calling it twice filters the rows, then the columns, and puts the image back the
 * right way round — so one simple, cache-friendly row loop does both directions.
 */
function filterRowsTransposed(src: Float64Array, w: number, h: number, kernel: Float64Array): Float64Array {
  const r = (kernel.length - 1) >> 1;
  const line = new Float64Array(w + 2 * r); // one row plus r mirrored pixels on each side
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let i = -r; i < w + r; i++) line[i + r] = src[y * w + mirrorIndex(i, w)];
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = 0; k < kernel.length; k++) sum += kernel[k] * line[x + k];
      out[x * h + y] = sum;
    }
  }
  return out;
}

/**
 * Filter with a symmetric, odd-length 1-D kernel along the rows, then along the columns.
 * A separable 2-D kernel k(x)·k(y) costs 2·n multiplications per pixel instead of n².
 *
 * Borders use mirror padding (`mirrorIndex`): a flat image stays flat, and a fine pattern
 * touching the border mostly averages out. (Replicate padding would copy the edge pixel
 * into the whole outer half of a wide kernel, leaving a striped frame where fine
 * detail meets the border.) No padding is perfect: the mirrored copy is only a guess at
 * what lies beyond the edge, so the outermost pixels can keep a faint artefact.
 */
export function convolveSeparable(plane: Plane, kernel: ArrayLike<number>): Plane {
  const { width: w, height: h } = plane;
  const k = Float64Array.from(kernel);
  const columnsAsRows = filterRowsTransposed(plane.data, w, h, k); // h × w
  return { data: filterRowsTransposed(columnsAsRows, h, w, k), width: w, height: h };
}

/** Sampled 1-D Gaussian of radius ⌈3σ⌉ (covers 99.7 % of its weight), normalised to sum to 1. */
export function gaussianKernel1d(sigma: number): Float64Array {
  const r = Math.max(1, Math.ceil(3 * sigma));
  const k = new Float64Array(2 * r + 1);
  let total = 0;
  for (let i = -r; i <= r; i++) total += k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
  return k.map(v => v / total);
}

/** Gaussian blur of standard deviation `sigma` pixels (σ ≤ 0 returns a copy). */
export function gaussianBlur(plane: Plane, sigma: number): Plane {
  if (sigma <= 0) return { ...plane, data: plane.data.slice() };
  return convolveSeparable(plane, gaussianKernel1d(sigma));
}
