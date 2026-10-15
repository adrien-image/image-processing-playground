import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { psnr } from '../denoising/denoise';
import { blurImage, convolveFft, makePsf, psfSpectrum, richardsonLucy, wiener } from './deblur';

const W = 64, H = 64;
const image = Float64Array.from({ length: W * H }, (_, i) => {
  const x = i % W, y = Math.floor(i / W);
  return (x > 16 && x < 44 && y > 20 && y < 40 ? 200 : 40) + 30 * Math.sin(x / 3);
});

describe('makePsf', () => {
  it('normalises every kind to sum 1', () => {
    for (const psf of [makePsf('motion', 15, 9, 30), makePsf('gaussian', 15, 2), makePsf('defocus', 15, 4)]) {
      expect(psf.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    }
  });

  it('draws a horizontal motion blur as a single row', () => {
    const psf = makePsf('motion', 9, 5, 0);
    const rows = new Set(Array.from(psf.keys()).filter(i => psf[i] > 0).map(i => Math.floor(i / 9)));
    expect([...rows]).toEqual([4]);
  });
});

describe('deconvolution', () => {
  const Hs = psfSpectrum(makePsf('gaussian', 9, 1.2), 9, W, H);

  it('a centred delta PSF leaves the image unchanged', () => {
    const delta = new Float64Array(9 * 9);
    delta[40] = 1;
    convolveFft(image, psfSpectrum(delta, 9, W, H)).forEach((v, i) => expect(v).toBeCloseTo(image[i], 8));
  });

  it('without noise, the inverse filter undoes the blur', () => {
    const blurred = blurImage(image, Hs, 0, seededRandom(1));
    expect(psnr(image, wiener(blurred, Hs, 0))).toBeGreaterThan(60);
  });

  it('with noise, Wiener beats the inverse filter, and both Wiener and Richardson–Lucy improve on the blurred image', () => {
    const Hstrong = psfSpectrum(makePsf('gaussian', 13, 2), 13, W, H);
    const blurred = blurImage(image, Hstrong, 5, seededRandom(1));
    const inverse = psnr(image, wiener(blurred, Hstrong, 0));
    const w = psnr(image, wiener(blurred, Hstrong, 0.03));
    expect(inverse).toBeLessThan(0);              // the noise divided by tiny |H| swamps everything
    expect(w).toBeGreaterThan(psnr(image, blurred) + 1.5);
    expect(psnr(image, richardsonLucy(blurred, Hstrong, 15))).toBeGreaterThan(psnr(image, blurred) + 1.5);
  });
});
