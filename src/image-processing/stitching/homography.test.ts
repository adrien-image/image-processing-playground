import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import {
  applyHomography, cameraHomography, fitHomography, invert, matchPoints, multiply, ransacHomography, stitch, symmetricEigen, type Homography, type Point,
} from './homography';

const H_TRUE: Homography = cameraHomography(12, 90, 300, 100, 80);
const random = seededRandom(5);
const src: Point[] = Array.from({ length: 40 }, () => ({ x: random() * 200, y: random() * 160 }));
const dst = src.map(p => applyHomography(H_TRUE, p));

describe('symmetricEigen', () => {
  it('diagonalises a symmetric matrix', () => {
    const M = [4, 1, 2, 1, 3, 0, 2, 0, 5];
    const { values, vectors } = symmetricEigen(M, 3);
    for (let k = 0; k < 3; k++) {
      const v = [0, 1, 2].map(r => vectors[r * 3 + k]);
      const Mv = [0, 1, 2].map(r => M[r * 3] * v[0] + M[r * 3 + 1] * v[1] + M[r * 3 + 2] * v[2]);
      Mv.forEach((x, r) => expect(x).toBeCloseTo(values[k] * v[r], 9));
    }
  });
});

describe('fitHomography', () => {
  it('recovers an exact homography from 4 and from many matches', () => {
    for (const n of [4, 40]) {
      const H = fitHomography(src.slice(0, n), dst.slice(0, n))!;
      H.forEach((v, i) => expect(v).toBeCloseTo(H_TRUE[i], 6));
    }
  });

  it('invert and multiply agree', () => {
    multiply(H_TRUE, invert(H_TRUE)).forEach((v, i) => expect(v).toBeCloseTo(i % 4 === 0 ? 1 : 0, 9));
  });
});

describe('ransacHomography', () => {
  it('ignores 30% wrong matches', () => {
    const noisy = dst.map((p, i) => (i % 10 < 3 ? { x: random() * 300, y: random() * 200 } : { x: p.x + (random() - 0.5), y: p.y + (random() - 0.5) }));
    const r = ransacHomography(src, noisy, { threshold: 2 })!;
    expect(r.inlierCount).toBeGreaterThanOrEqual(26);
    for (const p of [{ x: 0, y: 0 }, { x: 200, y: 160 }]) {
      const a = applyHomography(r.H, p), b = applyHomography(H_TRUE, p);
      expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeLessThan(1.5);
    }
  });
});

describe('stitch', () => {
  it('makes a canvas covering both images', () => {
    const img = { data: new Uint8ClampedArray(20 * 10 * 4).fill(200), width: 20, height: 10 };
    const shifted = stitch(img, img, [1, 0, 15, 0, 1, 0, 0, 0, 1], 'feather');
    expect(shifted.width).toBe(35);
    expect(shifted.height).toBe(10);
    expect(shifted.data[(5 * 35 + 17) * 4]).toBeCloseTo(200, 0); // overlap of two equal images stays equal
  });
});

describe('matchPoints', () => {
  it('pairs up the same corners in a shifted copy', () => {
    const w = 60, h = 40, rnd = seededRandom(3);
    const big = Float64Array.from({ length: (w + 10) * h }, () => rnd() * 255);
    const crop = (dx: number) => Float64Array.from({ length: w * h }, (_, i) => big[Math.floor(i / w) * (w + 10) + (i % w) + dx]);
    const a = crop(0), b = crop(7);                                     // b(x) = a(x + 7)
    const pts = Array.from({ length: 20 }, () => ({ x: 15 + Math.floor(rnd() * 30), y: 8 + Math.floor(rnd() * 24) }));
    const m = matchPoints(a, pts, b, pts.map(p => ({ x: p.x - 7, y: p.y })), w, h);
    expect(m.a.length).toBeGreaterThanOrEqual(18);
    m.a.forEach((p, i) => expect(p.x - m.b[i].x).toBe(7));
  });
});
