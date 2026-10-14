import { describe, expect, it } from 'vitest';
import { binarize, dilate, erode, morphology } from './morphology';

const W = 20, H = 20;
function image(draw: (set: (x: number, y: number) => void) => void): Float64Array {
  const img = new Float64Array(W * H);
  draw((x, y) => { img[y * W + x] = 255; });
  return img;
}
const square = (x0: number, y0: number, s: number) => (set: (x: number, y: number) => void) => {
  for (let y = y0; y < y0 + s; y++) for (let x = x0; x < x0 + s; x++) set(x, y);
};
const count = (img: Float64Array) => img.filter(v => v === 255).length;

describe('morphology', () => {
  it('erosion shrinks and dilation grows a square by the kernel radius', () => {
    const img = image(square(5, 5, 8));
    expect(count(erode(img, W, H, 3))).toBe(6 * 6);
    expect(count(dilate(img, W, H, 3))).toBe(10 * 10);
  });

  it('opening removes specks smaller than the kernel but keeps big shapes', () => {
    const img = image(set => { square(5, 5, 8)(set); set(16, 2); });
    const opened = morphology(img, W, H, 'open', 3);
    expect(opened[2 * W + 16]).toBe(0);
    expect(count(opened)).toBe(64);
  });

  it('closing fills holes smaller than the kernel', () => {
    const img = image(square(5, 5, 8));
    img[9 * W + 9] = 0;
    const closed = morphology(img, W, H, 'close', 3);
    expect(closed[9 * W + 9]).toBe(255);
    expect(count(closed)).toBe(64);
  });

  it('separable result equals the brute-force 2-D minimum', () => {
    const img = Float64Array.from({ length: W * H }, (_, i) => (i * 97) % 251);
    const fast = erode(img, W, H, 5);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let m = Infinity;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const px = x + dx, py = y + dy;
        if (px >= 0 && px < W && py >= 0 && py < H) m = Math.min(m, img[py * W + px]);
      }
      expect(fast[y * W + x]).toBe(m);
    }
  });

  it('binarize thresholds at value > t', () => {
    expect(Array.from(binarize(Float64Array.from([10, 128, 129]), 128))).toEqual([0, 0, 255]);
  });
});
