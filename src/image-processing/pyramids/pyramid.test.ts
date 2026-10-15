import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { createPlane, maxAbsDifference, type Plane } from './plane';
import {
  alphaBlend, blend, blendLevelLimit, blendLevels, expand, feather, gaussianPyramid, hardCut, laplacianFromGaussian,
  laplacianLevels, laplacianPyramid, maxLevels, reconstruct, reduce,
} from './pyramid';
import { makeMask } from './demos';

const fromFn = (w: number, h: number, f: (x: number, y: number) => number): Plane => {
  const p = createPlane(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) p.data[y * w + x] = f(x, y);
  return p;
};
const noise = (w: number, h: number, seed: number): Plane => {
  const rnd = seededRandom(seed);
  return fromFn(w, h, () => rnd() * 255);
};

describe('reduce', () => {
  it('halves the size, rounding up for odd sizes', () => {
    const r = reduce(createPlane(7, 5));
    expect([r.width, r.height]).toEqual([4, 3]);
    const r2 = reduce(createPlane(8, 1));
    expect([r2.width, r2.height]).toEqual([4, 1]);
  });

  it('keeps a constant image constant', () => {
    for (const v of reduce(createPlane(9, 6, 77)).data) expect(v).toBeCloseTo(77, 12);
  });

  it('blurs with [1 4 6 4 1]/16 before dropping pixels', () => {
    const impulse = createPlane(9, 1);
    impulse.data[4] = 16;
    // Kept columns 0, 2, 4, 6, 8 see the impulse at offsets −4, −2, 0, 2, 4.
    expect(Array.from(reduce(impulse).data)).toEqual([0, 1, 6, 1, 0]);
  });
});

describe('expand', () => {
  it('keeps a constant image constant, for even and odd target sizes', () => {
    for (const [w, h] of [[8, 6], [7, 5]]) {
      const e = expand(createPlane(4, 3, 50), w, h);
      expect([e.width, e.height]).toEqual([w, h]);
      for (const v of e.data) expect(v).toBeCloseTo(50, 12);
    }
  });

  it('reproduces a linear ramp in the interior', () => {
    const small = fromFn(8, 8, (x, y) => 2 * x + y);
    const big = expand(small, 16, 16);
    // Output pixel 2i sits on input pixel i: value 2·(x/2) + (y/2).
    for (let y = 2; y < 14; y++) for (let x = 2; x < 14; x++) expect(big.data[y * 16 + x]).toBeCloseTo(x + y / 2, 9);
  });
});

describe('Gaussian and Laplacian pyramids', () => {
  it('has the requested levels with sizes ⌈w/2ⁱ⌉', () => {
    const g = gaussianPyramid(createPlane(37, 23), 5);
    expect(g.map(p => [p.width, p.height])).toEqual([[37, 23], [19, 12], [10, 6], [5, 3], [3, 2]]);
    expect(maxLevels(256, 256)).toBe(9);
    expect(maxLevels(37, 23)).toBe(6); // 23 → 12 → 6 → 3 → 2 → 1
    expect(maxLevels(1, 1)).toBe(1);
  });

  it('Laplacian levels match the Gaussian sizes; the last level is the smallest Gaussian', () => {
    const img = noise(37, 23, 1);
    const g = gaussianPyramid(img, 4);
    const l = laplacianPyramid(img, 4);
    expect(l.map(p => [p.width, p.height])).toEqual(g.map(p => [p.width, p.height]));
    expect(maxAbsDifference(l[3], g[3])).toBe(0);
  });

  it('a shallower pyramid is a slice of a deeper one, plus its own residual', () => {
    const img = noise(40, 30, 3);
    const g = gaussianPyramid(img, 6), l = laplacianFromGaussian(g);
    for (const n of [1, 3, 6]) {
      const direct = laplacianPyramid(img, n), sliced = laplacianLevels(g, l, n);
      expect(sliced.length).toBe(n);
      sliced.forEach((p, i) => expect(maxAbsDifference(p, direct[i])).toBe(0));
    }
  });

  it('a constant image has all-zero bands', () => {
    const l = laplacianPyramid(createPlane(16, 16, 90), 4);
    for (const band of l.slice(0, -1)) for (const v of band.data) expect(Math.abs(v)).toBeLessThan(1e-12);
  });

  it('reconstruct(laplacianPyramid(x)) gives x back (up to rounding)', () => {
    for (const [w, h, levels] of [[64, 64, 6], [37, 23, 5], [1, 9, 3], [50, 31, 1]]) {
      const img = noise(w, h, w + h);
      expect(maxAbsDifference(reconstruct(laplacianPyramid(img, levels)), img)).toBeLessThan(1e-9);
    }
  });
});

describe('blending', () => {
  const a = noise(33, 20, 7), b = noise(33, 20, 8);

  it('multiband blend with an all-ones mask returns a, all-zeros returns b', () => {
    expect(maxAbsDifference(blend(a, b, createPlane(33, 20, 1), 5), a)).toBeLessThan(1e-9);
    expect(maxAbsDifference(blend(a, b, createPlane(33, 20, 0), 5), b)).toBeLessThan(1e-9);
  });

  it('feather and hard cut obey the same limits', () => {
    for (const fn of [(m: Plane) => feather(a, b, m, 4), (m: Plane) => hardCut(a, b, m)]) {
      expect(maxAbsDifference(fn(createPlane(33, 20, 1)), a)).toBeLessThan(1e-9);
      expect(maxAbsDifference(fn(createPlane(33, 20, 0)), b)).toBeLessThan(1e-9);
    }
  });

  it('one level is a hard cut for a binary mask, a plain alpha blend for a soft one', () => {
    const mask = makeMask('vertical', 33, 20);
    expect(maxAbsDifference(blend(a, b, mask, 1), hardCut(a, b, mask))).toBeLessThan(1e-9);
    const flatA = createPlane(4, 1, 200), flatB = createPlane(4, 1, 0);
    const soft: Plane = { data: Float64Array.from([0.3, 0.5, 0.7, 0.9]), width: 4, height: 1 };
    expect(Array.from(blend(flatA, flatB, soft, 1).data)).toEqual([60, 100, 140, 180]);
    expect(Array.from(hardCut(flatA, flatB, soft).data)).toEqual([0, 200, 200, 200]);
  });

  describe('blendLevelLimit', () => {
    // Blending a flat A = 1 with a flat B = 0 gives the effective weight of A at every pixel.
    const N = 256;
    const weightOfA = (mask: Plane, levels: number) => blend(createPlane(N, N, 1), createPlane(N, N, 0), mask, levels);
    const at = (p: Plane, x: number, y: number) => p.data[y * p.width + x];

    it('stops while the coarsest level is still at least 16 px', () => {
      expect(blendLevelLimit(256, 256)).toBe(5); // 256, 128, 64, 32, 16
      expect(blendLevelLimit(256, 160)).toBe(4); // 160, 80, 40, 20
      expect(blendLevelLimit(31, 40)).toBe(2); // 31, 16
      expect(blendLevelLimit(10, 300)).toBe(1);
      expect(blendLevelLimit(256, 256, 2)).toBe(maxLevels(256, 256) - 1);
    });

    it('at the limit, each side keeps its own image and the disc stays centred', () => {
      const n = blendLevelLimit(N, N);
      const split = weightOfA(makeMask('vertical', N, N), n);
      expect(at(split, 0, 128)).toBeGreaterThan(0.999);
      expect(at(split, 255, 128)).toBeLessThan(0.001);
      const disc = weightOfA(makeMask('disc', N, N), n);
      expect(at(disc, 128, 128)).toBeGreaterThan(0.999);
      for (const [x, y] of [[0, 0], [255, 0], [0, 255], [255, 255], [0, 128], [255, 128]]) expect(at(disc, x, y)).toBeLessThan(0.001);
      let asymmetry = 0; // point symmetry about the centre
      for (let i = 0; i < N * N; i++) asymmetry = Math.max(asymmetry, Math.abs(disc.data[i] - disc.data[N * N - 1 - i]));
      expect(asymmetry).toBeLessThan(0.01);
    });

    it('deeper pyramids leak A into B, unevenly (why the slider stops at the limit)', () => {
      const split = weightOfA(makeMask('vertical', N, N), 8); // 2 × 2 residual
      expect(at(split, 255, 128)).toBeGreaterThan(0.3); // far right should be pure B (0)
      expect(at(split, 0, 128)).toBeLessThan(0.95); // far left should be pure A (1)
      const disc = weightOfA(makeMask('disc', N, N), 8);
      expect(at(disc, 128, 128)).toBeLessThan(0.5); // the disc centre is no longer A
      expect(at(disc, 255, 255) - at(disc, 0, 0)).toBeGreaterThan(0.2); // lopsided: grid anchored top-left
    });
  });

  it('the transition widens with more levels', () => {
    const flatA = createPlane(64, 8, 200), flatB = createPlane(64, 8, 0);
    const mask = makeMask('vertical', 64, 8);
    const width = (p: Plane) => p.data.slice(0, 64).filter(v => v > 5 && v < 195).length; // columns in transition
    expect(width(blend(flatA, flatB, mask, 5))).toBeGreaterThan(width(blend(flatA, flatB, mask, 2)));
    expect(width(blend(flatA, flatB, mask, 2))).toBeGreaterThan(0);
  });

  it('a shallower blend can reuse the bands of a deeper one (only the residual changes)', () => {
    const mask = makeMask('disc', 33, 20);
    const ga = gaussianPyramid(a, 5), gb = gaussianPyramid(b, 5), gm = gaussianPyramid(mask, 5);
    const deepBands = blendLevels(laplacianFromGaussian(ga), laplacianFromGaussian(gb), gm);
    for (const n of [1, 3, 5]) {
      const residual = alphaBlend(ga[n - 1], gb[n - 1], gm[n - 1]);
      const viaSlice = reconstruct([...deepBands.slice(0, n - 1), residual]);
      expect(maxAbsDifference(viaSlice, blend(a, b, mask, n))).toBe(0);
    }
  });

  it('blendLevels mixes each band with the matching mask level', () => {
    const la = laplacianPyramid(a, 3), lb = laplacianPyramid(b, 3);
    const half = gaussianPyramid(createPlane(33, 20, 0.5), 3);
    const mixed = blendLevels(la, lb, half);
    expect(maxAbsDifference(mixed[1], alphaBlend(la[1], lb[1], half[1]))).toBe(0);
    expect(mixed[1].data[0]).toBeCloseTo((la[1].data[0] + lb[1].data[0]) / 2, 12);
  });
});
