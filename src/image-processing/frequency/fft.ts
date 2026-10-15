/**
 * 2D FFT for image frequency analysis: iterative radix-2 Cooley–Tukey on separate
 * real/imaginary Float64Arrays (no per-butterfly allocations). Dimensions must be
 * powers of two — see `padToPowerOfTwo`.
 */

/** A complex-valued image stored as two planes. */
export interface ComplexImage {
  re: Float64Array;
  im: Float64Array;
  width: number;
  height: number;
}

/** In-place 1-D FFT of (re, im); `inverse` computes the unscaled inverse transform. */
export function fft1d(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) throw new Error(`FFT length must be a power of two (got ${n})`);

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  // Butterflies
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (sign * 2 * Math.PI) / len;
    const wRe = Math.cos(angle), wIm = Math.sin(angle);
    const half = len >> 1;
    for (let start = 0; start < n; start += len) {
      let cRe = 1, cIm = 0;
      for (let k = 0; k < half; k++) {
        const a = start + k, b = a + half;
        const vRe = re[b] * cRe - im[b] * cIm;
        const vIm = re[b] * cIm + im[b] * cRe;
        re[b] = re[a] - vRe; im[b] = im[a] - vIm;
        re[a] += vRe; im[a] += vIm;
        const next = cRe * wRe - cIm * wIm;
        cIm = cRe * wIm + cIm * wRe;
        cRe = next;
      }
    }
  }
}

function transform2d(img: ComplexImage, inverse: boolean): ComplexImage {
  const { width: w, height: h } = img;
  const re = new Float64Array(img.re), im = new Float64Array(img.im);

  const rowRe = new Float64Array(w), rowIm = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    rowRe.set(re.subarray(y * w, (y + 1) * w));
    rowIm.set(im.subarray(y * w, (y + 1) * w));
    fft1d(rowRe, rowIm, inverse);
    re.set(rowRe, y * w);
    im.set(rowIm, y * w);
  }

  const colRe = new Float64Array(h), colIm = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) { colRe[y] = re[y * w + x]; colIm[y] = im[y * w + x]; }
    fft1d(colRe, colIm, inverse);
    for (let y = 0; y < h; y++) { re[y * w + x] = colRe[y]; im[y * w + x] = colIm[y]; }
  }

  if (inverse) {
    const n = w * h;
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  return { re, im, width: w, height: h };
}

/** Forward 2-D FFT (rows, then columns). */
export const fft2d = (img: ComplexImage) => transform2d(img, false);
/** Inverse 2-D FFT, scaled by 1/N so ifft2d(fft2d(x)) = x. */
export const ifft2d = (img: ComplexImage) => transform2d(img, true);

/** Smallest power of two ≥ n. */
export function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Place a real image in a power-of-two buffer. The padding is filled with the
 * image's mean so the border doesn't add a strong artificial edge to the spectrum.
 */
export function padToPowerOfTwo(pixels: Float64Array, width: number, height: number): ComplexImage {
  const pw = nextPowerOf2(width), ph = nextPowerOf2(height);
  let mean = 0;
  for (let i = 0; i < pixels.length; i++) mean += pixels[i];
  mean /= pixels.length || 1;
  const re = new Float64Array(pw * ph).fill(mean);
  for (let y = 0; y < height; y++) re.set(pixels.subarray(y * width, (y + 1) * width), y * pw);
  return { re, im: new Float64Array(pw * ph), width: pw, height: ph };
}

/** Real part of the top-left width×height region (undoes `padToPowerOfTwo`). */
export function cropReal(img: ComplexImage, width: number, height: number): Float64Array {
  const out = new Float64Array(width * height);
  for (let y = 0; y < height; y++) out.set(img.re.subarray(y * img.width, y * img.width + width), y * width);
  return out;
}

/** Index of (x, y) after swapping quadrants so the DC term sits at the centre. */
const shiftedIndex = (x: number, y: number, w: number, h: number) =>
  ((y + (h >> 1)) % h) * w + ((x + (w >> 1)) % w);

/** log(1 + |F|) scaled to 0–255, quadrant-shifted so low frequencies are in the middle. */
export function magnitudeSpectrum(freq: ComplexImage): Float64Array {
  const { width: w, height: h } = freq;
  const out = new Float64Array(w * h);
  let max = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = shiftedIndex(x, y, w, h);
      const v = Math.log1p(Math.hypot(freq.re[i], freq.im[i]));
      out[y * w + x] = v;
      if (v > max) max = v;
    }
  }
  if (max > 0) for (let i = 0; i < out.length; i++) out[i] *= 255 / max;
  return out;
}

export type FilterKind = 'ideal' | 'gaussian';
export type FilterType = 'low' | 'high' | 'band-pass' | 'band-stop';

/**
 * Gain (0–1) for every frequency, in the *centred* layout (DC in the middle) so it
 * can be displayed directly. `radius` is a distance in frequency bins from DC;
 * band filters use [inner, outer] (the order of the two radii doesn't matter).
 * For Gaussian filters the radius is the σ of the Gaussian.
 */
export function filterMask(w: number, h: number, type: FilterType, kind: FilterKind, r1: number, r2 = r1): Float64Array {
  const inner = Math.min(r1, r2), outer = Math.max(r1, r2);
  const mask = new Float64Array(w * h);
  const lowpass = (d: number, r: number) =>
    kind === 'ideal' ? (d <= r ? 1 : 0) : Math.exp(-(d * d) / (2 * r * r));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const d = Math.hypot(x - (w >> 1), y - (h >> 1));
      let g: number;
      switch (type) {
        case 'low': g = lowpass(d, r1); break;
        case 'high': g = 1 - lowpass(d, r1); break;
        case 'band-pass': g = Math.max(0, lowpass(d, outer) - (kind === 'ideal' ? lowpass(d, inner - 1e-9) : lowpass(d, inner))); break;
        case 'band-stop': g = Math.min(1, 1 - lowpass(d, outer) + (kind === 'ideal' ? lowpass(d, inner - 1e-9) : lowpass(d, inner))); break;
      }
      mask[y * w + x] = g;
    }
  }
  return mask;
}

/** Multiply a spectrum (standard, uncentred layout) by a centred gain mask. */
export function applyMask(freq: ComplexImage, mask: Float64Array): ComplexImage {
  const { width: w, height: h } = freq;
  const re = new Float64Array(freq.re), im = new Float64Array(freq.im);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = shiftedIndex(x, y, w, h);
      const g = mask[y * w + x];
      re[i] *= g;
      im[i] *= g;
    }
  }
  return { re, im, width: w, height: h };
}
