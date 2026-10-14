import { describe, expect, it } from 'vitest';
import { applyMask, cropReal, fft1d, fft2d, filterMask, ifft2d, magnitudeSpectrum, padToPowerOfTwo, type ComplexImage } from './fft';

const real = (re: number[], w: number, h: number): ComplexImage =>
  ({ re: Float64Array.from(re), im: new Float64Array(re.length), width: w, height: h });

describe('fft', () => {
  it('matches a direct DFT', () => {
    const x = [1, 2, 0, -1, 3, 0.5, 2, 1];
    const re = Float64Array.from(x), im = new Float64Array(8);
    fft1d(re, im);
    for (let k = 0; k < 8; k++) {
      let sr = 0, si = 0;
      for (let n = 0; n < 8; n++) {
        sr += x[n] * Math.cos((-2 * Math.PI * k * n) / 8);
        si += x[n] * Math.sin((-2 * Math.PI * k * n) / 8);
      }
      expect(re[k]).toBeCloseTo(sr, 10);
      expect(im[k]).toBeCloseTo(si, 10);
    }
  });

  it('round-trips through ifft2d', () => {
    const px = Array.from({ length: 64 }, (_, i) => (i * 13) % 17);
    const back = ifft2d(fft2d(real(px, 8, 8)));
    px.forEach((v, i) => expect(back.re[i]).toBeCloseTo(v, 9));
  });

  it('puts the DC term (sum of pixels) at index 0 and the centre of the spectrum', () => {
    const f = fft2d(real(Array(16).fill(3), 4, 4));
    expect(f.re[0]).toBeCloseTo(48, 10);
    const spec = magnitudeSpectrum(f);
    expect(spec[2 * 4 + 2]).toBeCloseTo(255, 9);
  });

  it('rejects non power-of-two sizes', () => {
    expect(() => fft1d(new Float64Array(6), new Float64Array(6))).toThrow(/power of two/);
  });
});

describe('filters', () => {
  it('a wide ideal low-pass is the identity', () => {
    const px = Array.from({ length: 64 }, (_, i) => (i * 7) % 11);
    const f = fft2d(real(px, 8, 8));
    const out = ifft2d(applyMask(f, filterMask(8, 8, 'low', 'ideal', 100)));
    px.forEach((v, i) => expect(out.re[i]).toBeCloseTo(v, 9));
  });

  it('high-pass removes the mean (DC)', () => {
    const f = fft2d(real(Array.from({ length: 64 }, (_, i) => 100 + (i % 3)), 8, 8));
    const out = ifft2d(applyMask(f, filterMask(8, 8, 'high', 'ideal', 1)));
    expect(out.re.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 8);
  });

  it('band filters accept the radii in either order', () => {
    expect(filterMask(16, 16, 'band-pass', 'ideal', 6, 2)).toEqual(filterMask(16, 16, 'band-pass', 'ideal', 2, 6));
    expect(Math.max(...filterMask(16, 16, 'band-pass', 'gaussian', 5, 2))).toBeGreaterThan(0);
  });

  it('band-pass + band-stop = 1 for ideal filters', () => {
    const p = filterMask(16, 16, 'band-pass', 'ideal', 2, 6);
    const s = filterMask(16, 16, 'band-stop', 'ideal', 2, 6);
    p.forEach((v, i) => expect(v + s[i]).toBeCloseTo(1, 12));
  });
});

describe('padding', () => {
  it('pads with the mean and crops back', () => {
    const padded = padToPowerOfTwo(Float64Array.from([1, 2, 3, 4, 5, 6]), 3, 2);
    expect(padded.width).toBe(4);
    expect(padded.re[3]).toBe(3.5);
    expect(Array.from(cropReal(padded, 3, 2))).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
