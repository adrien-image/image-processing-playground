/**
 * Undoing a blur (deconvolution).
 *
 * A blurred photo is the sharp image convolved with the blur kernel (the point spread
 * function, PSF: what a single point of light becomes), plus noise:  g = h ⊛ f + n.
 * By the convolution theorem that is a product in the frequency domain: G = H · F + N.
 *
 *   inverse filter   F̂ = G / H          exact without noise, but wherever H ≈ 0 the noise
 *                                         N / H explodes
 *   Wiener filter    F̂ = H* · G / (|H|² + K)   K (noise-to-signal ratio) stops the division
 *                                         from blowing up where the blur destroyed the signal
 *   Richardson–Lucy  an iteration that keeps the image positive and stays close to the
 *                    data: f ← f · (h̃ ⊛ (g / (h ⊛ f))), h̃ = h flipped (astronomy, microscopy)
 *
 * Everything uses circular (wrap-around) convolution through the FFT, so the image size
 * must be a power of two. Real photos need their borders tapered first.
 */
import { fft2d, ifft2d, type ComplexImage } from '../frequency/fft';
import { gaussianRandom } from '../../lib/random';

export type PsfKind = 'motion' | 'gaussian' | 'defocus';

/** A normalised PSF in a size×size square (size odd), centred. */
export function makePsf(kind: PsfKind, size: number, param: number, angleDeg = 0): Float64Array {
  const psf = new Float64Array(size * size);
  const c = (size - 1) / 2;
  if (kind === 'motion') {
    // A line of length `param` at `angleDeg`, drawn with sub-pixel samples.
    const a = (angleDeg * Math.PI) / 180;
    const samples = Math.max(2, Math.round(param * 8));
    for (let s = 0; s < samples; s++) {
      const t = (s / (samples - 1) - 0.5) * param;
      const x = Math.round(c + t * Math.cos(a)), y = Math.round(c - t * Math.sin(a));
      if (x >= 0 && y >= 0 && x < size && y < size) psf[y * size + x] += 1;
    }
  } else {
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const d2 = (x - c) ** 2 + (y - c) ** 2;
      psf[y * size + x] = kind === 'gaussian' ? Math.exp(-d2 / (2 * param * param)) : d2 <= param * param ? 1 : 0;
    }
  }
  const sum = psf.reduce((a, b) => a + b, 0) || 1;
  return psf.map(v => v / sum);
}

/** The PSF's frequency response H for a w×h image: the PSF placed with its centre at (0, 0), wrapping around. */
export function psfSpectrum(psf: Float64Array, size: number, w: number, h: number): ComplexImage {
  const re = new Float64Array(w * h);
  const c = (size - 1) / 2;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const px = (x - c + w) % w, py = (y - c + h) % h;
    re[py * w + px] += psf[y * size + x];
  }
  return fft2d({ re, im: new Float64Array(w * h), width: w, height: h });
}

const spectrum = (pixels: Float64Array, w: number, h: number) => fft2d({ re: Float64Array.from(pixels), im: new Float64Array(w * h), width: w, height: h });

/** Circular convolution of an image with a PSF (via its spectrum H). */
export function convolveFft(pixels: Float64Array, H: ComplexImage): Float64Array {
  const { width: w, height: h } = H;
  const F = spectrum(pixels, w, h);
  const re = new Float64Array(w * h), im = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    re[i] = F.re[i] * H.re[i] - F.im[i] * H.im[i];
    im[i] = F.re[i] * H.im[i] + F.im[i] * H.re[i];
  }
  return ifft2d({ re, im, width: w, height: h }).re;
}

/** Blur with the PSF, then add Gaussian noise of standard deviation `sigma` (no clamping). */
export function blurImage(pixels: Float64Array, H: ComplexImage, sigma: number, random: () => number): Float64Array {
  const out = convolveFft(pixels, H);
  if (sigma > 0) for (let i = 0; i < out.length; i++) out[i] += sigma * gaussianRandom(random);
  return out;
}

/**
 * Wiener deconvolution F̂ = H*·G / (|H|² + K). K = 0 is the plain inverse filter
 * (with a tiny floor so exact zeros of H don't divide by zero).
 */
export function wiener(blurred: Float64Array, H: ComplexImage, K: number): Float64Array {
  const { width: w, height: h } = H;
  const G = spectrum(blurred, w, h);
  const re = new Float64Array(w * h), im = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const hr = H.re[i], hi = H.im[i];
    const den = hr * hr + hi * hi + Math.max(K, 1e-12);
    // H* · G = (hr − i·hi)(gr + i·gi)
    re[i] = (hr * G.re[i] + hi * G.im[i]) / den;
    im[i] = (hr * G.im[i] - hi * G.re[i]) / den;
  }
  return ifft2d({ re, im, width: w, height: h }).re;
}

/** Richardson–Lucy deconvolution; `onIteration` sees every intermediate estimate. */
export function richardsonLucy(blurred: Float64Array, H: ComplexImage, iterations: number): Float64Array {
  const { width: w, height: h } = H;
  // The flipped PSF's spectrum is the complex conjugate of H.
  const Hflip: ComplexImage = { re: H.re, im: H.im.map(v => -v), width: w, height: h };
  const offset = 1; // keep every value strictly positive
  const g = blurred.map(v => Math.max(0, v) + offset);
  let f = new Float64Array(w * h).fill(g.reduce((a, b) => a + b, 0) / g.length);
  for (let k = 0; k < iterations; k++) {
    const reblurred = convolveFft(f, H);
    const ratio = g.map((v, i) => v / Math.max(reblurred[i], 1e-6));
    const correction = convolveFft(ratio, Hflip);
    f = f.map((v, i) => v * correction[i]);
  }
  return f.map(v => v - offset);
}

/** log(1 + |H|) normalised to 0–1, quadrant-shifted (DC in the centre), for display. */
export function responseImage(H: ComplexImage): Float64Array {
  const { width: w, height: h } = H;
  const out = new Float64Array(w * h);
  let max = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = ((y + h / 2) % h) * w + ((x + w / 2) % w);
    const v = Math.hypot(H.re[i], H.im[i]);
    out[y * w + x] = v;
    max = Math.max(max, v);
  }
  return out.map(v => v / (max || 1));
}
