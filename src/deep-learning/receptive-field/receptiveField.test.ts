import { describe, expect, it } from 'vitest';
import { dependencyRange, fieldRange, layerGeometry, pathCounts, PRESETS, type LayerSpec } from './receptiveField';

describe('layerGeometry', () => {
  it('matches the sizes of LeNet-5', () => {
    const g = layerGeometry(28, PRESETS.lenet.layers);
    expect(g.map(l => l.size)).toEqual([28, 24, 12, 8, 4]);
    expect(g.map(l => l.field)).toEqual([1, 5, 6, 14, 16]);
    expect(g.map(l => l.jump)).toEqual([1, 1, 2, 2, 4]);
  });

  it('grows the field by 2 per padded 3×3 layer, and exponentially with dilation', () => {
    expect(layerGeometry(32, PRESETS.stacked.layers).map(l => l.field)).toEqual([1, 3, 5, 7, 9, 11]);
    expect(layerGeometry(48, PRESETS.dilated.layers).map(l => l.field)).toEqual([1, 3, 7, 15, 31]);
  });

  it('agrees with brute-force dependency tracking', () => {
    const layers: LayerSpec[] = [
      { kind: 'conv', kernel: 3, stride: 2, padding: 1, dilation: 1 },
      { kind: 'conv', kernel: 3, stride: 1, padding: 2, dilation: 2 },
      { kind: 'pool', kernel: 2, stride: 2, padding: 0, dilation: 1 },
    ];
    const g = layerGeometry(40, layers);
    const last = g.length - 1;
    for (let i = 0; i < g[last].size; i++) {
      const counts = pathCounts(layers, g, last, i);
      const touched = [...counts.keys()].filter(x => counts[x] > 0);
      const [lo, hi] = fieldRange(g[last], i);
      // Every input pixel that has a path lies inside the theoretical field…
      touched.forEach(x => { expect(x).toBeGreaterThanOrEqual(lo); expect(x).toBeLessThanOrEqual(hi); });
      // …and the clipped dependency range is exactly the span of those pixels.
      expect(dependencyRange(layers, g, last, [i, i], 0)).toEqual([Math.min(...touched), Math.max(...touched)]);
    }
  });
});

describe('pathCounts', () => {
  it('is peaked in the middle: a 1-D binomial-like profile', () => {
    const layers = PRESETS.stacked.layers;
    const g = layerGeometry(32, layers);
    const counts = pathCounts(layers, g, layers.length, 16);
    // Five 3×3 layers: 3^5 paths in total, spread as the coefficients of (1 + x + x²)^5.
    expect(counts.reduce((a, b) => a + b, 0)).toBe(3 ** 5);
    expect(Array.from(counts.slice(11, 22))).toEqual([1, 5, 15, 30, 45, 51, 45, 30, 15, 5, 1]);
  });
});
