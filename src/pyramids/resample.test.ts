import { describe, expect, it } from 'vitest';
import { createPlane, type Plane } from './plane';
import {
  antiAliasSigma, blurThenSubsample, crop, keysCubic, resize, resizeAntialiased, subsample, type ResizeMethod,
} from './resample';

const fromFn = (w: number, h: number, f: (x: number, y: number) => number): Plane => {
  const p = createPlane(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) p.data[y * w + x] = f(x, y);
  return p;
};
const at = (p: Plane, x: number, y: number) => p.data[y * p.width + x];
const METHODS: ResizeMethod[] = ['nearest', 'bilinear', 'bicubic'];

/** Input coordinate of output pixel i under pixel-centre alignment. */
const source = (i: number, inSize: number, outSize: number) => (i + 0.5) * (inSize / outSize) - 0.5;

describe('keysCubic', () => {
  it('interpolates (1 at 0, 0 at other integers) and its weights sum to 1', () => {
    expect(keysCubic(0)).toBe(1);
    for (const t of [1, 2, -1, -2, 3]) expect(keysCubic(t)).toBeCloseTo(0, 12);
    for (const f of [0, 0.25, 0.5, 0.9]) {
      expect(keysCubic(1 + f) + keysCubic(f) + keysCubic(1 - f) + keysCubic(2 - f)).toBeCloseTo(1, 12);
    }
  });
});

describe('resize', () => {
  it('nearest ×2 duplicates every pixel into a 2×2 block', () => {
    const img = fromFn(5, 3, (x, y) => x * 10 + y);
    const out = resize(img, 10, 6, 'nearest');
    for (let y = 0; y < 6; y++) for (let x = 0; x < 10; x++) expect(at(out, x, y)).toBe(at(img, x >> 1, y >> 1));
  });

  it('bilinear keeps a linear ramp linear in the interior (non-integer scale)', () => {
    const f = (x: number, y: number) => 3 * x - 2 * y + 50;
    const out = resize(fromFn(16, 12, f), 40, 30, 'bilinear');
    for (let y = 0; y < 30; y++) {
      for (let x = 0; x < 40; x++) {
        const sx = source(x, 16, 40), sy = source(y, 12, 30);
        if (sx < 0 || sx > 15 || sy < 0 || sy > 11) continue; // borders are clamped
        expect(at(out, x, y)).toBeCloseTo(f(sx, sy), 9);
      }
    }
  });

  it('bicubic reproduces a linear ramp in the interior', () => {
    const f = (x: number, y: number) => 1.5 * x + 4 * y + 7;
    const out = resize(fromFn(16, 16, f), 37, 37, 'bicubic');
    let checked = 0;
    for (let y = 0; y < 37; y++) {
      for (let x = 0; x < 37; x++) {
        const sx = source(x, 16, 37), sy = source(y, 16, 37);
        if (sx < 1 || sx > 14 || sy < 1 || sy > 14) continue; // all 4 taps inside the image
        expect(at(out, x, y)).toBeCloseTo(f(sx, sy), 9);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(500);
  });

  it('keeps a constant image constant when shrinking or enlarging, for every method', () => {
    for (const method of METHODS) {
      for (const [w, h] of [[7, 5], [40, 33]]) {
        for (const v of resize(createPlane(20, 15, 99), w, h, method).data) expect(v).toBeCloseTo(99, 9);
      }
    }
  });

  it('returns the input unchanged at scale 1', () => {
    const img = fromFn(6, 4, (x, y) => (x * 7 + y * 13) % 11);
    for (const method of METHODS) expect(Array.from(resize(img, 6, 4, method).data)).toEqual(Array.from(img.data));
  });
});

describe('crop', () => {
  it('copies the requested rectangle', () => {
    const img = fromFn(6, 5, (x, y) => x + 10 * y);
    const c = crop(img, 2, 1, 3, 2);
    expect(Array.from(c.data)).toEqual([12, 13, 14, 22, 23, 24]);
  });
});

describe('subsampling and aliasing', () => {
  // Columns alternate 0 / 255: a 2-pixel period, exactly the Nyquist frequency.
  const stripes = fromFn(32, 8, x => (x % 2 ? 255 : 0));

  it('keeps every factor-th pixel, ⌈w/f⌉ × ⌈h/f⌉', () => {
    const img = fromFn(9, 7, (x, y) => x + 100 * y);
    const s = subsample(img, 4);
    expect([s.width, s.height]).toEqual([3, 2]);
    expect(Array.from(s.data)).toEqual([0, 4, 8, 400, 404, 408]);
  });

  it('naive subsampling turns the stripes into a flat (wrong) image', () => {
    const s = subsample(stripes, 2);
    expect(s.width).toBe(16);
    expect(new Set(s.data).size).toBe(1); // aliased to a constant: the pattern vanished into black
    expect(s.data[0]).toBe(0);
  });

  it('blur-then-subsample gives the mean gray instead', () => {
    const s = blurThenSubsample(stripes, 2);
    for (const v of s.data) expect(Math.abs(v - 127.5)).toBeLessThan(3);
  });

  it('chooses a stronger blur for bigger factors', () => {
    expect(antiAliasSigma(4)).toBeGreaterThan(antiAliasSigma(2));
    const s = blurThenSubsample(fromFn(64, 64, x => (Math.floor(x / 2) % 2 ? 255 : 0)), 4); // 4-px period, ÷4
    for (let x = 0; x < s.width; x++) expect(Math.abs(at(s, x, 8) - 127.5)).toBeLessThan(40);
  });
});

describe('resizeAntialiased', () => {
  it('matches any target size and keeps constants constant', () => {
    for (const [w, h] of [[10, 30], [64, 7]]) {
      const out = resizeAntialiased(createPlane(25, 20, 140), w, h);
      expect([out.width, out.height]).toEqual([w, h]);
      for (const v of out.data) expect(v).toBeCloseTo(140, 9);
    }
  });

  it('removes stripes that plain bilinear shrinking would alias', () => {
    const stripes = fromFn(64, 4, x => (x % 2 ? 255 : 0));
    const spread = (p: Plane) => Math.max(...p.data.slice(4, 20)) - Math.min(...p.data.slice(4, 20));
    expect(spread(resizeAntialiased(stripes, 24, 4))).toBeLessThan(20);
    expect(spread(resize(stripes, 24, 4, 'bilinear'))).toBeGreaterThan(50);
  });
});
