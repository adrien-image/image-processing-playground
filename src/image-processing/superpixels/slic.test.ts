import { describe, expect, it } from 'vitest';
import { boundaries, boundaryRecall, enforceConnectivity, rgbToLab, slic } from './slic';

const W = 64, H = 48;
/** A wavy two-colour split plus a disc: shapes that a regular grid cannot follow. */
function picture() {
  const rgba = new Uint8ClampedArray(W * H * 4);
  const region = new Int32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    region[i] = (x - 40) ** 2 + (y - 24) ** 2 < 100 ? 2 : y < 20 + 6 * Math.sin(x / 5) ? 1 : 0;
    const c = [[200, 60, 50], [40, 90, 200], [240, 220, 60]][region[i]];
    rgba.set([...c, 255], i * 4);
  }
  return { rgba, edges: boundaries(region, W, H) };
}

describe('rgbToLab', () => {
  it('matches reference values', () => {
    const lab = rgbToLab(new Uint8ClampedArray([255, 255, 255, 255, 255, 0, 0, 255]), 2);
    expect(lab.L[0]).toBeCloseTo(100, 1);
    expect(Math.abs(lab.a[0]) + Math.abs(lab.b[0])).toBeLessThan(0.1);
    expect(lab.L[1]).toBeCloseTo(53.24, 1);
    expect(lab.a[1]).toBeCloseTo(80.09, 0);
    expect(lab.b[1]).toBeCloseTo(67.2, 0);
  });
});

describe('slic', () => {
  it('gives about K connected regions that follow the edges', () => {
    const { rgba, edges } = picture();
    const r = slic(rgba, W, H, 48, 10, 10);
    expect(r.count).toBeGreaterThan(30);
    expect(r.count).toBeLessThan(70);
    const again = enforceConnectivity(r.labels, W, H, 1);
    expect(again.count).toBe(r.count);                   // already one piece per label
    expect(boundaryRecall(edges, boundaries(r.labels, W, H), W, H)).toBeGreaterThan(0.9);
  });

  it('high compactness gives a grid that ignores the edges', () => {
    const { rgba, edges } = picture();
    const recall = (m: number) => boundaryRecall(edges, boundaries(slic(rgba, W, H, 12, m, 10).labels, W, H), W, H);
    expect(recall(200)).toBeLessThan(recall(5) - 0.15);
  });
});
