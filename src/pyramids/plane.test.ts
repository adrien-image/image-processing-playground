import { describe, expect, it } from 'vitest';
import {
  addPlanes, convolveSeparable, createPlane, gaussianBlur, gaussianKernel1d, maxAbsDifference, mergeRGB, mirrorIndex,
  splitRGB, subtractPlanes, type Plane,
} from './plane';
import { makeMask, seaScene, sunsetScene, zonePlate } from './demos';

const fromFn = (w: number, h: number, f: (x: number, y: number) => number): Plane => {
  const p = createPlane(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) p.data[y * w + x] = f(x, y);
  return p;
};

describe('splitRGB / mergeRGB', () => {
  it('round-trips RGBA bytes (alpha becomes opaque)', () => {
    const rgba = new Uint8ClampedArray([10, 20, 30, 40, 250, 0, 128, 255]);
    const planes = splitRGB(rgba, 2, 1);
    expect(planes.map(p => Array.from(p.data))).toEqual([[10, 250], [20, 0], [30, 128]]);
    expect(Array.from(mergeRGB(planes))).toEqual([10, 20, 30, 255, 250, 0, 128, 255]);
  });

  it('shows one plane as gray and applies offset/gain with clamping', () => {
    const p: Plane = { data: Float64Array.from([-10, 0, 300]), width: 3, height: 1 };
    expect(Array.from(mergeRGB([p], 128, 2))).toEqual([108, 108, 108, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
  });
});

describe('plane arithmetic', () => {
  it('adds, subtracts and measures differences', () => {
    const a = fromFn(3, 2, (x, y) => x + 10 * y);
    const b = fromFn(3, 2, () => 1);
    expect(maxAbsDifference(subtractPlanes(addPlanes(a, b), b), a)).toBe(0);
    expect(maxAbsDifference(a, addPlanes(a, b))).toBe(1);
  });
});

describe('mirrorIndex', () => {
  it('reflects around the edge pixels without repeating them', () => {
    expect([-3, -2, -1, 0, 1, 4, 5, 6, 7].map(i => mirrorIndex(i, 5))).toEqual([3, 2, 1, 0, 1, 4, 3, 2, 1]);
    expect(mirrorIndex(-12, 3)).toBe(0); // bounces several times for kernels wider than the image
    expect(mirrorIndex(7, 1)).toBe(0);
  });
});

describe('gaussian blur', () => {
  it('uses a normalised kernel of radius ⌈3σ⌉', () => {
    const k = gaussianKernel1d(2);
    expect(k.length).toBe(13);
    expect(k.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
  });

  it('keeps a constant image constant, including the borders', () => {
    const out = gaussianBlur(createPlane(9, 7, 42), 3);
    for (const v of out.data) expect(v).toBeCloseTo(42, 9);
  });

  it('averages fine stripes to gray right up to the border', () => {
    const stripes = fromFn(16, 16, (_, y) => (y % 2 ? 255 : 0));
    const out = gaussianBlur(stripes, 2);
    for (let x = 0; x < 16; x++) expect(Math.abs(out.data[x] - 127.5)).toBeLessThan(2); // top row
  });

  it('keeps a linear ramp unchanged away from the borders (symmetric kernel)', () => {
    const ramp = fromFn(20, 20, (x, y) => 3 * x + y);
    const out = convolveSeparable(ramp, [0.25, 0.5, 0.25]);
    expect(out.data[10 * 20 + 10]).toBeCloseTo(40, 9);
  });
});

describe('demo images', () => {
  it('zone plate is white in the centre and stays in 0–255', () => {
    const z = zonePlate(64);
    expect(z.data[32 * 64 + 32]).toBeGreaterThan(254);
    expect(Math.min(...z.data)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...z.data)).toBeLessThanOrEqual(255);
  });

  it('colour scenes are opaque RGBA of the requested size', () => {
    for (const img of [sunsetScene(16, 12), seaScene(16, 12)]) {
      expect(img.length).toBe(16 * 12 * 4);
      expect(img[3]).toBe(255);
    }
  });

  it('masks are binary and split the image in two', () => {
    const v = makeMask('vertical', 10, 4);
    expect(v.data[0]).toBe(1);
    expect(v.data[9]).toBe(0);
    expect(v.data.reduce((s, x) => s + x, 0)).toBe(20);
    expect(makeMask('horizontal', 4, 10).data[39]).toBe(0);
    const d = makeMask('disc', 20, 20);
    expect(d.data[10 * 20 + 10]).toBe(1);
    expect(d.data[0]).toBe(0);
  });
});
