/**
 * Resampling: changing the number of pixels of an image.
 *
 * - `resize` interpolates between pixels (nearest / bilinear / bicubic).
 * - `subsample` keeps every n-th pixel — fast, but it aliases.
 * - `blurThenSubsample` removes the frequencies the smaller image cannot hold first.
 */
import { createPlane, gaussianBlur, type Plane } from './plane';

export type ResizeMethod = 'nearest' | 'bilinear' | 'bicubic';

/**
 * Keys' cubic convolution kernel. With a = −0.5 it is the Catmull-Rom spline (the "bicubic"
 * of Pillow and MATLAB's imresize; OpenCV uses a = −0.75, which is a little sharper).
 * It is 1 at t = 0, 0 at every other integer (so original samples are kept), spans
 * 4 pixels, and reproduces any quadratic exactly. It is *not* the cubic through all four
 * neighbours: the curve passes through the two middle pixels, and the outer two only set
 * its slopes there — so neighbouring pieces join smoothly.
 */
export function keysCubic(t: number, a = -0.5): number {
  const x = Math.abs(t);
  if (x <= 1) return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1;
  if (x < 2) return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a;
  return 0;
}

/** For each output position along one axis: which input pixels to read and with what weights. */
interface AxisTaps {
  /** Pixels read per output position. */
  taps: number;
  /** `index[i * taps + k]`: k-th input pixel for output position i (already clamped to the image). */
  index: Int32Array;
  weight: Float64Array;
}

/**
 * Pixel-centre alignment: pixel i covers [i, i + 1) and its centre is i + 0.5. Output
 * centre i' + 0.5 maps to input coordinate (i' + 0.5) · in/out, i.e. input pixel
 * x = (i' + 0.5) · in/out − 0.5. The image's outer edges line up exactly, whatever the
 * scale (the convention of OpenCV and of `align_corners = false` in deep-learning libraries).
 */
function axisTaps(inSize: number, outSize: number, method: ResizeMethod): AxisTaps {
  const scale = inSize / outSize;
  const taps = method === 'nearest' ? 1 : method === 'bilinear' ? 2 : 4;
  const index = new Int32Array(outSize * taps);
  const weight = new Float64Array(outSize * taps);
  const clampIdx = (i: number) => Math.min(inSize - 1, Math.max(0, i)); // clamped borders
  for (let i = 0; i < outSize; i++) {
    const x = (i + 0.5) * scale - 0.5;
    if (method === 'nearest') {
      index[i] = clampIdx(Math.floor(x + 0.5)); // the closest input centre
      weight[i] = 1;
    } else if (method === 'bilinear') {
      const x0 = Math.floor(x), f = x - x0; // straight line between the 2 neighbours
      index[i * 2] = clampIdx(x0); weight[i * 2] = 1 - f;
      index[i * 2 + 1] = clampIdx(x0 + 1); weight[i * 2 + 1] = f;
    } else {
      const x0 = Math.floor(x); // Keys kernel over the 4 neighbours x0−1 … x0+2
      for (let k = 0; k < 4; k++) {
        index[i * 4 + k] = clampIdx(x0 - 1 + k);
        weight[i * 4 + k] = keysCubic(x - (x0 - 1 + k));
      }
    }
  }
  return { taps, index, weight };
}

/**
 * Resize a plane to `newWidth × newHeight` with pixel-centre alignment (see `axisTaps`).
 * All three methods are separable: resize every row horizontally, then every column
 * vertically. Pixels beyond the border are clamped to the edge. Bicubic may overshoot
 * slightly below 0 or above 255 next to sharp edges (values are not clamped here).
 *
 * There is no anti-alias filter: shrinking a lot with this function aliases just like
 * `subsample` — blur first (see `blurThenSubsample`) or use a pyramid.
 */
export function resize(plane: Plane, newWidth: number, newHeight: number, method: ResizeMethod): Plane {
  const { width: w, height: h, data } = plane;
  const tx = axisTaps(w, newWidth, method);
  const ty = axisTaps(h, newHeight, method);

  // Horizontal pass: w × h → newWidth × h.
  const rows = new Float64Array(newWidth * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < newWidth; x++) {
      let sum = 0;
      for (let k = 0; k < tx.taps; k++) sum += tx.weight[x * tx.taps + k] * data[y * w + tx.index[x * tx.taps + k]];
      rows[y * newWidth + x] = sum;
    }
  }
  // Vertical pass: newWidth × h → newWidth × newHeight.
  const out = createPlane(newWidth, newHeight);
  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      let sum = 0;
      for (let k = 0; k < ty.taps; k++) sum += ty.weight[y * ty.taps + k] * rows[ty.index[y * ty.taps + k] * newWidth + x];
      out.data[y * newWidth + x] = sum;
    }
  }
  return out;
}

/** Copy the `width × height` rectangle whose top-left corner is (x0, y0). */
export function crop(plane: Plane, x0: number, y0: number, width: number, height: number): Plane {
  const out = createPlane(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) out.data[y * width + x] = plane.data[(y0 + y) * plane.width + x0 + x];
  }
  return out;
}

/**
 * Naive downsampling: keep pixels 0, f, 2f, … in each direction (⌈w/f⌉ × ⌈h/f⌉ result).
 * Detail finer than 2f pixels cannot be represented at the new spacing and shows up
 * as false, coarser patterns instead (aliasing / moiré).
 */
export function subsample(plane: Plane, factor: number): Plane {
  const w = Math.ceil(plane.width / factor), h = Math.ceil(plane.height / factor);
  const out = createPlane(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) out.data[y * w + x] = plane.data[y * factor * plane.width + x * factor];
  }
  return out;
}

/**
 * σ of the anti-alias Gaussian for a reduction by `factor`: σ = factor / 2.
 * After subsampling, the highest frequency left (Nyquist) is 1/(2·factor) cycles/pixel.
 * A Gaussian's frequency response is exp(−2π²σ²ν²): with this σ it is ≈ 0.29 at the new
 * Nyquist and falls quickly above it, so most of what would alias is gone while most of
 * the detail the small image can show is kept.
 *
 * It is a compromise. A Gaussian is a soft low-pass: frequencies just above the limit
 * still get through weakly and fold back as faint false patterns. A larger σ removes more
 * of them but blurs the result more; an ideal "brick-wall" filter (a sinc) would need an
 * infinitely wide kernel and makes ripples next to edges.
 */
export function antiAliasSigma(factor: number): number {
  return factor / 2;
}

/** Low-pass filter (Gaussian, σ = factor/2), then keep every `factor`-th pixel. */
export function blurThenSubsample(plane: Plane, factor: number): Plane {
  return subsample(gaussianBlur(plane, antiAliasSigma(factor)), factor);
}

/**
 * Resize for display or for matching two images' sizes: bilinear, preceded by the
 * anti-alias blur when the image shrinks (σ from the larger of the two shrink factors).
 */
export function resizeAntialiased(plane: Plane, newWidth: number, newHeight: number): Plane {
  const shrink = Math.max(plane.width / newWidth, plane.height / newHeight);
  const source = shrink > 1 ? gaussianBlur(plane, antiAliasSigma(shrink)) : plane;
  return resize(source, newWidth, newHeight, 'bilinear');
}
