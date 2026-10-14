import { describe, expect, it } from 'vitest';
import { addGaussianNoise, addSaltPepper, gaussianFilter, medianFilter, psnr } from './denoise';
import { seededRandom } from '../lib/random';

const W = 32, H = 32;
const flat = (v: number) => new Float64Array(W * H).fill(v);

describe('medianFilter', () => {
  it('removes isolated salt-and-pepper pixels completely', () => {
    const img = flat(120);
    img[5 * W + 5] = 255;
    img[20 * W + 9] = 0;
    expect(Array.from(medianFilter(img, W, H, 1)).every(v => v === 120)).toBe(true);
  });

  it('has no dark border (replicate padding)', () => {
    const out = medianFilter(flat(200), W, H, 3);
    expect(out[0]).toBe(200);
    expect(out[W * H - 1]).toBe(200);
  });

  it('beats the Gaussian on salt-and-pepper noise, measured by PSNR', () => {
    const clean = Float64Array.from({ length: W * H }, (_, i) => ((i % W) < 16 ? 60 : 200));
    const noisy = addSaltPepper(clean, 0.1, seededRandom(5));
    expect(psnr(clean, medianFilter(noisy, W, H, 1))).toBeGreaterThan(psnr(clean, gaussianFilter(noisy, W, H, 1)) + 5);
  });
});

describe('gaussianFilter', () => {
  it('keeps flat regions flat, including the border', () => {
    for (const v of gaussianFilter(flat(90), W, H, 2)) expect(v).toBeCloseTo(90, 9);
  });

  it('improves PSNR on Gaussian noise', () => {
    const clean = flat(128);
    const noisy = addGaussianNoise(clean, 20, seededRandom(9));
    expect(psnr(clean, gaussianFilter(noisy, W, H, 2))).toBeGreaterThan(psnr(clean, noisy));
  });
});

describe('noise', () => {
  it('is reproducible for a seed and respects the density', () => {
    const a = addSaltPepper(flat(100), 0.2, seededRandom(1));
    const b = addSaltPepper(flat(100), 0.2, seededRandom(1));
    expect(Array.from(a)).toEqual(Array.from(b));
    const changed = a.filter(v => v !== 100).length / a.length;
    expect(changed).toBeGreaterThan(0.12);
    expect(changed).toBeLessThan(0.28);
  });

  it('psnr is infinite for identical images', () => {
    expect(psnr(flat(3), flat(3))).toBe(Infinity);
  });
});
