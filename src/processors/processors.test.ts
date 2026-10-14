import { describe, expect, it } from 'vitest';
import { convolve, makeOutputMapper, normaliseRange, sampleClamped } from './utils';
import { gaussianKernel, sharpenKernel, SOBEL_X } from './kernels';
import { applyProcessor, defaultConfig } from './apply';
import { SharpenProcessor } from './sharpen';
import { GaussianBlurProcessor } from './gaussian-blur';
import { SobelXProcessor } from './sobel';
import { getAllProcessors, initProcessors } from '.';

const flat = (w: number, h: number, v: number) => new Float64Array(w * h).fill(v);

/** 32×32 checkerboard with 8 px tiles, white 255 / dark 51 (same as the demo image). */
function checkerboard(): Float64Array {
  const out = new Float64Array(32 * 32);
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++)
      out[y * 32 + x] = ((x >> 3) + (y >> 3)) % 2 === 0 ? 255 : 51;
  return out;
}

describe('convolve', () => {
  it('uses replicate padding, so a flat image has no edges — even at the border', () => {
    const g = convolve(flat(16, 16, 51), 16, 16, SOBEL_X);
    expect(Math.max(...g.map(Math.abs))).toBe(0);
  });

  it('keeps a flat image unchanged under a normalised blur', () => {
    const g = convolve(flat(8, 8, 100), 8, 8, gaussianKernel(1, 1));
    for (const v of g) expect(v).toBeCloseTo(100, 10);
  });

  it('does not flip the kernel (cross-correlation)', () => {
    // A single bright pixel reproduces the kernel rotated 180°.
    const img = new Float64Array(9);
    img[4] = 1;
    const k = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    expect(Array.from(convolve(img, 3, 3, k))).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1]);
  });
});

describe('sampleClamped', () => {
  it('clamps coordinates to the nearest edge', () => {
    const img = Float64Array.from([1, 2, 3, 4]);
    expect(sampleClamped(img, 2, 2, -5, -5)).toBe(1);
    expect(sampleClamped(img, 2, 2, 9, 0)).toBe(2);
    expect(sampleClamped(img, 2, 2, 1, 9)).toBe(4);
  });
});

describe('makeOutputMapper', () => {
  it('magnitude: scales the strongest |response| to 255', () => {
    const map = makeOutputMapper(Float64Array.from([-40, 10, 20]), 'magnitude');
    expect(map(-40)).toBe(255);
    expect(map(20)).toBe(128);
  });

  it('clamp: leaves in-range values alone', () => {
    const map = makeOutputMapper(Float64Array.from([765]), 'clamp');
    expect(map(255)).toBe(255);
    expect(map(51)).toBe(51);
    expect(map(-3)).toBe(0);
    expect(map(765)).toBe(255);
  });
});

describe('normaliseRange', () => {
  it('maps min→0 and max→255', () => {
    expect(Array.from(normaliseRange(Float64Array.from([-10, 0, 10])))).toEqual([0, 127.5, 255]);
  });
});

describe('kernels', () => {
  it('gaussian sums to 1 and σ≈0.85 matches the classic [1 2 1]/16 kernel', () => {
    const k = gaussianKernel(1, 0.85);
    expect(k.flat().reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(k[1][1]).toBeCloseTo(4 / 16, 2);
    expect(k[0][0]).toBeCloseTo(1 / 16, 2);
  });

  it('sharpen weights sum to 1 and amount 1 is the classic kernel', () => {
    expect(sharpenKernel(1)).toEqual([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]);
    expect(sharpenKernel(2.5).flat().reduce((a, b) => a + b, 0)).toBe(1);
  });
});

describe('processors', () => {
  it('sharpen keeps flat white areas white (regression: was scaled down to 85)', () => {
    const img = checkerboard();
    const { output } = applyProcessor(SharpenProcessor, img, 32, 32);
    expect(output[3 * 32 + 3]).toBe(255); // inside a white tile
    expect(output[3 * 32 + 11]).toBe(51); // inside a dark tile
  });

  it('gaussian blur preserves flat regions', () => {
    const { output } = applyProcessor(GaussianBlurProcessor, checkerboard(), 32, 32);
    expect(output[3 * 32 + 3]).toBe(255);
    expect(output[0]).toBe(255); // corner: replicate padding, no dark halo
  });

  it('sobel responds on tile boundaries only', () => {
    const { output } = applyProcessor(SobelXProcessor, checkerboard(), 32, 32);
    expect(output[3 * 32 + 3]).toBe(0);
    expect(output[3 * 32 + 7]).toBe(255);
    expect(output[3 * 32 + 0]).toBe(0); // no frame artefact at the left border
  });

  it('defaultConfig reads each param default', () => {
    expect(defaultConfig(SharpenProcessor)).toEqual({ amount: 1 });
    expect(defaultConfig(SobelXProcessor)).toEqual({});
  });

  it('initProcessors registers each built-in once and is idempotent', () => {
    initProcessors();
    initProcessors();
    expect(getAllProcessors().map(p => p.id)).toEqual(['sobel-x', 'sobel-y', 'gaussian-blur', 'sharpen']);
  });

  it('every processor exposes an odd square kernel', () => {
    initProcessors();
    for (const p of getAllProcessors()) {
      const k = p.kernel(defaultConfig(p));
      expect(k.length % 2).toBe(1);
      for (const row of k) expect(row.length).toBe(k.length);
    }
  });
});
