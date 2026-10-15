import { describe, expect, it } from 'vitest';
import { diffusionInpaint, exemplarInpaint, holeError } from './inpaint';

const W = 48, H = 40;
function image(colour: (x: number, y: number) => number) {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = colour(x, y);
    data.set([v, v, v, 255], (y * W + x) * 4);
  }
  return { data, width: W, height: H };
}
const square = new Uint8Array(W * H).map((_, i) => (Math.abs((i % W) - 24) < 6 && Math.abs(Math.floor(i / W) - 20) < 6 ? 1 : 0));

describe('diffusionInpaint', () => {
  it('fills a smooth ramp almost exactly', () => {
    const img = image((x, y) => 40 + 3 * x + 2 * y);
    expect(holeError(img.data, diffusionInpaint(img, square, 600), square)).toBeLessThan(1);
  });
});

describe('exemplarInpaint', () => {
  const stripes = image(x => (x % 6 < 3 ? 220 : 30));

  it('continues a texture that diffusion blurs away', () => {
    const ex = exemplarInpaint(stripes, square, 3, 30);
    const diffusion = holeError(stripes.data, diffusionInpaint(stripes, square, 600), square);
    const exemplar = holeError(stripes.data, ex.data, square);
    expect(diffusion).toBeGreaterThan(60);
    expect(exemplar).toBeLessThan(10);
  });

  it('records when each pixel was filled', () => {
    const { filledAt, steps } = exemplarInpaint(stripes, square, 3, 30);
    square.forEach((m, i) => expect(filledAt[i] > 0).toBe(m === 1));
    expect(steps.length).toBeGreaterThan(2);
    expect(Math.max(...filledAt)).toBeLessThanOrEqual(steps.length);
  });
});
