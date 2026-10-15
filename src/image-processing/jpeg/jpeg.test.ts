import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { psnr } from '../denoising/denoise';
import { category, dct2, estimateBits, idct2, jpeg, LUMA_QUANT, quantTable, ZIGZAG } from './jpeg';

const random = seededRandom(3);
const block = Float64Array.from({ length: 64 }, () => random() * 255 - 128);

describe('DCT', () => {
  it('is orthonormal: inverse recovers the block and energy is preserved', () => {
    const c = dct2(block);
    idct2(c).forEach((v, i) => expect(v).toBeCloseTo(block[i], 9));
    const energy = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v * v, 0);
    expect(energy(c)).toBeCloseTo(energy(block), 6);
  });

  it('puts a flat block entirely into the DC coefficient (8 × the value)', () => {
    const c = dct2(new Float64Array(64).fill(10));
    expect(c[0]).toBeCloseTo(80, 9);
    for (let i = 1; i < 64; i++) expect(Math.abs(c[i])).toBeLessThan(1e-9);
  });
});

describe('quantTable / ZIGZAG / category', () => {
  it('quality 50 is the standard table; higher quality means smaller steps', () => {
    expect(quantTable(50)).toEqual(LUMA_QUANT);
    expect(quantTable(100).every(v => v === 1)).toBe(true);
    quantTable(90).forEach((v, i) => expect(v).toBeLessThanOrEqual(quantTable(20)[i]));
  });

  it('zigzag visits every position once, starting 0, 1, 8, 16, 9, 2', () => {
    expect(new Set(ZIGZAG).size).toBe(64);
    expect(ZIGZAG.slice(0, 6)).toEqual([0, 1, 8, 16, 9, 2]);
    expect(ZIGZAG[63]).toBe(63);
  });

  it('size categories', () => {
    expect([0, 1, -1, 2, 3, 4, -7, 8, 255].map(category)).toEqual([0, 1, 1, 2, 2, 3, 3, 4, 8]);
  });
});

describe('jpeg', () => {
  const w = 40, h = 36; // not multiples of 8
  const image = Float64Array.from({ length: w * h }, (_, i) => 128 + 80 * Math.sin((i % w) / 5) * Math.cos(Math.floor(i / w) / 7));

  it('trades quality for size', () => {
    const high = jpeg(image, w, h, 90), low = jpeg(image, w, h, 10);
    expect(high.output).toHaveLength(w * h);
    expect(psnr(image, high.output)).toBeGreaterThan(psnr(image, low.output));
    expect(psnr(image, high.output)).toBeGreaterThan(35);
    expect(estimateBits(low)).toBeLessThan(estimateBits(high));
    expect(low.nonZero).toBeLessThan(high.nonZero);
  });

  it('a flat image costs about one symbol per block', () => {
    const flat = jpeg(new Float64Array(64 * 64).fill(200), 64, 64, 50);
    flat.output.forEach(v => expect(v).toBe(200));
    expect(estimateBits(flat)).toBeLessThan(64 * 3);
  });
});
