import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { applyTps, bendingEnergy, fitTps, solveLinear, warpImage, type Point } from './tps';

const random = seededRandom(3);
const src: Point[] = [
  { x: 0.1, y: 0.1 }, { x: 0.9, y: 0.1 }, { x: 0.1, y: 0.9 }, { x: 0.9, y: 0.9 },
  { x: 0.5, y: 0.5 }, { x: 0.3, y: 0.7 }, { x: 0.7, y: 0.35 },
];
const affine = (p: Point): Point => ({ x: 0.1 + 1.2 * p.x - 0.3 * p.y, y: -0.05 + 0.2 * p.x + 0.9 * p.y });

describe('solveLinear', () => {
  it('solves a small system that needs pivoting', () => {
    const x = solveLinear([[0, 2, 1], [1, 1, 0], [3, 0, 1]], [7, 3, 6]);
    [1, 2, 3].forEach((v, i) => expect(x[i]).toBeCloseTo(v, 12));
  });
});

describe('fitTps', () => {
  it('interpolates the control points exactly when λ = 0', () => {
    const dst = src.map(p => ({ x: p.x + 0.1 * (random() - 0.5), y: p.y + 0.1 * (random() - 0.5) }));
    const t = fitTps(src, dst);
    src.forEach((p, i) => {
      const q = applyTps(t, p);
      expect(q.x).toBeCloseTo(dst[i].x, 10);
      expect(q.y).toBeCloseTo(dst[i].y, 10);
    });
  });

  it('reproduces an affine map exactly, with no bending', () => {
    const t = fitTps(src, src.map(affine));
    expect(bendingEnergy(t)).toBeCloseTo(0, 10);
    for (const p of [{ x: 0.2, y: 0.4 }, { x: 0.95, y: 0.05 }, { x: -0.3, y: 1.4 }]) {
      const q = applyTps(t, p), a = affine(p);
      expect(q.x).toBeCloseTo(a.x, 10);
      expect(q.y).toBeCloseTo(a.y, 10);
    }
  });

  it('bends less as λ grows, tending to the best affine fit', () => {
    const dst = src.map((p, i) => (i === 4 ? { x: 0.65, y: 0.6 } : p)); // pull the centre point
    const energies = [0, 0.01, 1, 100].map(l => bendingEnergy(fitTps(src, dst, l)));
    for (let i = 1; i < energies.length; i++) expect(energies[i]).toBeLessThan(energies[i - 1]);
    expect(energies[3]).toBeLessThan(1e-3 * energies[0]);
    const t = fitTps(src, dst, 1e6);
    const q = applyTps(t, src[4]);
    expect(Math.hypot(q.x - 0.65, q.y - 0.6)).toBeGreaterThan(0.1); // no longer reaches the moved point
  });
});

describe('warpImage', () => {
  it('leaves the image unchanged under the identity warp', () => {
    const w = 12, h = 10;
    const rgba = Uint8ClampedArray.from({ length: w * h * 4 }, (_, i) => (i % 4 === 3 ? 255 : (i * 37) % 256));
    const out = warpImage(rgba, w, h, fitTps(src, src));
    for (let i = 0; i < rgba.length; i++) expect(Math.abs(out[i] - rgba[i])).toBeLessThanOrEqual(1);
  });
});
