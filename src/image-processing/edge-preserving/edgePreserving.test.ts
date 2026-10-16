import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { addGaussianNoise, gaussianFilter, psnr } from '../denoising/denoise';
import { bilateralFilter, bilateralWeights, nlmWeights, nonLocalMeans } from './edgePreserving';

const W = 48;
/** A vertical step edge (60 | 190) with stripes on the right half. */
const clean = Float64Array.from({ length: W * W }, (_, i) => {
  const x = i % W, y = Math.floor(i / W);
  return (x < W / 2 ? 60 : 190) + (x > 30 && (y >> 2) % 2 ? 30 : 0);
});
const flat = new Float64Array(W * W).fill(100);

describe('bilateralFilter', () => {
  it('has weights that sum to 1 and keeps flat images flat', () => {
    expect(bilateralWeights(clean, W, W, 10, 10, 3, 2, 30).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (const v of bilateralFilter(flat, W, W, 2, 2, 20)) expect(v).toBeCloseTo(100, 10);
  });

  it('gives almost no weight to pixels across an edge', () => {
    const weights = bilateralWeights(clean, W, W, W / 2 - 1, 20, 3, 2, 20); // just left of the edge
    let across = 0;
    for (let dy = 0; dy < 7; dy++) for (let dx = 4; dx < 7; dx++) across += weights[dy * 7 + dx];
    expect(across).toBeLessThan(1e-6);
  });

  it('keeps the edge sharp where a Gaussian blurs it', () => {
    const row = 20 * W;
    const bil = bilateralFilter(clean, W, W, 3, 2, 20), gauss = gaussianFilter(clean, W, W, 3);
    expect(Math.abs(bil[row + W / 2 - 1] - 60)).toBeLessThan(1);
    expect(Math.abs(gauss[row + W / 2 - 1] - 60)).toBeGreaterThan(20);
  });

  it('becomes a Gaussian blur when σr is very large', () => {
    const noisy = addGaussianNoise(clean, 15, seededRandom(1));
    const wide = bilateralFilter(noisy, W, W, 3, 1.4, 1e6);
    const gauss = gaussianFilter(noisy, W, W, 3);
    // denoise.ts picks σ = 1.4 for a 7×7 window
    for (let i = 0; i < wide.length; i += 37) expect(wide[i]).toBeCloseTo(gauss[i], 6);
  });
});

describe('nonLocalMeans', () => {
  const sigma = 20;
  const noisy = addGaussianNoise(clean, sigma, seededRandom(3));
  const opts = { patchRadius: 1, searchRadius: 5, sigma, h: 0.8 * sigma };

  it('has weights that sum to 1, largest on similar patches', () => {
    const weights = nlmWeights(clean, W, W, 10, 20, opts);
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    const centre = weights[5 * 11 + 5];
    // A patch in the same flat region weighs as much as the centre; one across the edge does not.
    expect(nlmWeights(clean, W, W, W / 2 - 3, 20, opts)[5 * 11 + 10]).toBeLessThan(1e-6);
    expect(weights[5 * 11 + 8]).toBeCloseTo(centre, 12);
  });

  it('removes more noise than a Gaussian or the bilateral filter, and keeps flat images flat', () => {
    const nlm = psnr(clean, nonLocalMeans(noisy, W, W, opts));
    expect(nlm).toBeGreaterThan(psnr(clean, gaussianFilter(noisy, W, W, 2)) + 1);
    expect(nlm).toBeGreaterThan(psnr(clean, bilateralFilter(noisy, W, W, 3, 2, 2 * sigma)));
    for (const v of nonLocalMeans(flat, W, W, opts)) expect(v).toBeCloseTo(100, 10);
  });
});
