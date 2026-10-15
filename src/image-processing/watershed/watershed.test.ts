import { describe, expect, it } from 'vitest';
import { labelComponents } from '../regions/components';
import { distanceTransform, findMarkers, rasteriseDiscs, watershed } from './watershed';

const W = 80, H = 50;
const twoCoins = rasteriseDiscs([{ x: 25, y: 25, r: 14 }, { x: 50, y: 25, r: 14 }], W, H);

describe('distanceTransform', () => {
  it('is the exact Euclidean distance to the background', () => {
    const mask = new Uint8Array(W * H).fill(1);
    mask[10 * W + 10] = 0;
    const d = distanceTransform(mask, W, H);
    for (const [x, y] of [[13, 14], [40, 30], [10, 10], [0, 49]]) expect(d[y * W + x]).toBeCloseTo(Math.hypot(x - 10, y - 10), 9);
  });

  it('peaks at the coin centres', () => {
    const d = distanceTransform(twoCoins, W, H);
    expect(d[25 * W + 25]).toBeGreaterThan(13);
    expect(d[25 * W + 37]).toBeLessThan(d[25 * W + 25] - 4);   // the saddle where they touch
  });
});

describe('watershed', () => {
  it('splits two touching coins that connected components see as one', () => {
    expect(labelComponents(twoCoins, W, H, 4).count).toBe(1);
    const d = distanceTransform(twoCoins, W, H);
    const { markers, count } = findMarkers(d, W, H, 5, 3);
    expect(count).toBe(2);
    const r = watershed(d.map(v => -v), markers, twoCoins, W, H);
    expect(r.labels[25 * W + 20]).not.toBe(r.labels[25 * W + 55]);
    for (let y = 20; y <= 30; y++) expect(r.lines[y * W + 37] + r.lines[y * W + 38]).toBeGreaterThan(0); // dam on the midline
    let covered = 0, fg = 0;
    twoCoins.forEach((v, i) => { if (v) { fg++; if (r.labels[i] || r.lines[i]) covered++; } });
    expect(covered).toBe(fg);
    expect(r.labels.every((l, i) => l === 0 || twoCoins[i] === 1)).toBe(true);
  });
});
