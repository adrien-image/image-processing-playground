import { describe, expect, it } from 'vitest';
import { labelColour, labelComponents, regionProperties, removeSmall, type Connectivity } from './components';
import { seededRandom } from '../lib/random';

/** Build a mask from ASCII art: '#' = foreground, anything else = background. */
function maskFrom(rows: string[]): { mask: Uint8Array; width: number; height: number } {
  const width = rows[0].length;
  const height = rows.length;
  const mask = new Uint8Array(width * height);
  rows.forEach((row, y) => [...row].forEach((c, x) => { mask[y * width + x] = c === '#' ? 1 : 0; }));
  return { mask, width, height };
}

/** Filled disk of radius r centred at (cx, cy), using pixel centres. */
function diskMask(size: number, cx: number, cy: number, r: number): Uint8Array {
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r) mask[y * size + x] = 1;
    }
  }
  return mask;
}

/** Independent reference: union–find over neighbour pairs, renumbered by first appearance. */
function referenceLabels(mask: Uint8Array, w: number, h: number, connectivity: Connectivity): Int32Array {
  const parent = Int32Array.from({ length: w * h }, (_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const union = (a: number, b: number) => { parent[find(a)] = find(b); };
  const back = connectivity === 4 ? [[-1, 0], [0, -1]] : [[-1, 0], [0, -1], [-1, -1], [1, -1]];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (const [dx, dy] of back) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < w && ny >= 0 && mask[ny * w + nx]) union(y * w + x, ny * w + nx);
      }
    }
  }
  const out = new Int32Array(w * h);
  const rootLabel = new Map<number, number>();
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) continue;
    const root = find(i);
    if (!rootLabel.has(root)) rootLabel.set(root, rootLabel.size + 1);
    out[i] = rootLabel.get(root)!;
  }
  return out;
}

describe('labelComponents', () => {
  it('counts the separate blobs of a synthetic image', () => {
    const { mask, width, height } = maskFrom([
      '##....#....',
      '##...###...',
      '......#....',
      '...........',
      '#.......###',
      '#.......###',
    ]);
    expect(labelComponents(mask, width, height, 8).count).toBe(4);
    expect(labelComponents(mask, width, height, 4).count).toBe(4);
  });

  it('two squares touching only at a corner: 2 regions with 4-connectivity, 1 with 8', () => {
    const { mask, width, height } = maskFrom([
      '###.....',
      '###.....',
      '###.....',
      '...###..',
      '...###..',
      '...###..',
    ]);
    expect(labelComponents(mask, width, height, 4).count).toBe(2);
    expect(labelComponents(mask, width, height, 8).count).toBe(1);
  });

  it('a ring with a hole is a single component, and the hole stays background', () => {
    const size = 41;
    const outer = diskMask(size, 20.5, 20.5, 16);
    const inner = diskMask(size, 20.5, 20.5, 9);
    const ring = outer.map((v, i) => (v && !inner[i] ? 1 : 0));
    for (const c of [4, 8] as const) {
      const { labels, count } = labelComponents(ring, size, size, c);
      expect(count).toBe(1);
      expect(labels[20 * size + 20]).toBe(0); // centre of the hole
    }
  });

  it('numbers regions 1..count in raster order of their first pixel', () => {
    const { mask, width, height } = maskFrom([
      '....##',
      '#...##',
      '#.....',
      '...#..',
    ]);
    const { labels, count } = labelComponents(mask, width, height, 8);
    expect(count).toBe(3);
    expect(labels[4]).toBe(1); // first pixel of the top-right square
    expect(labels[1 * width]).toBe(2); // left bar starts on row 1
    expect(labels[3 * width + 3]).toBe(3);
  });

  it('labels are contiguous 1..count and match an independent union–find on random masks', () => {
    const w = 37, h = 23;
    for (const seed of [1, 2, 3, 4, 5]) {
      const random = seededRandom(seed);
      const mask = Uint8Array.from({ length: w * h }, () => (random() < 0.45 ? 1 : 0));
      for (const c of [4, 8] as const) {
        const { labels, count } = labelComponents(mask, w, h, c);
        expect(Array.from(labels)).toEqual(Array.from(referenceLabels(mask, w, h, c)));
        expect(new Set(labels.filter(l => l > 0))).toEqual(new Set(Array.from({ length: count }, (_, i) => i + 1)));
      }
    }
  });

  it('does not connect the end of one row to the start of the next', () => {
    const { mask, width, height } = maskFrom([
      '...#',
      '#...',
    ]);
    expect(labelComponents(mask, width, height, 8).count).toBe(2);
  });

  it('returns 0 regions for an empty image', () => {
    const { labels, count } = labelComponents(new Uint8Array(20 * 10), 20, 10, 8);
    expect(count).toBe(0);
    expect(labels.every(l => l === 0)).toBe(true);
    expect(regionProperties(labels, count, 20, 10)).toEqual([]);
  });

  it('handles a fully foreground image as one region', () => {
    const mask = new Uint8Array(30 * 20).fill(1);
    expect(labelComponents(mask, 30, 20, 4).count).toBe(1);
  });
});

describe('regionProperties', () => {
  it('gives exact area, centroid (pixel centres) and inclusive bbox', () => {
    const { mask, width, height } = maskFrom([
      '..........',
      '..###.....',
      '..###.....',
      '.......#..',
      '.......#..',
      '......##..',
    ]);
    const { labels, count } = labelComponents(mask, width, height, 8);
    const [rect, ell] = regionProperties(labels, count, width, height);

    expect(rect.area).toBe(6);
    expect(rect.centroid).toEqual({ x: 3.5, y: 2 }); // x ∈ {2,3,4} → 3 + 0.5; y ∈ {1,2} → 1.5 + 0.5
    expect(rect.bbox).toEqual({ x0: 2, y0: 1, x1: 4, y1: 2 });

    expect(ell.area).toBe(4);
    // Pixels (7,3) (7,4) (6,5) (7,5): mean x = 27/4 + 0.5, mean y = 17/4 + 0.5
    expect(ell.centroid.x).toBeCloseTo(7.25, 12);
    expect(ell.centroid.y).toBeCloseTo(4.75, 12);
    expect(ell.bbox).toEqual({ x0: 6, y0: 3, x1: 7, y1: 5 });
  });

  it('perimeter counts region pixels with a 4-neighbour outside the region or the image', () => {
    // 5×5 square in the middle: only its 3×3 core is interior → 25 − 9 = 16 boundary pixels.
    const { mask, width, height } = maskFrom([
      '.......',
      '.#####.',
      '.#####.',
      '.#####.',
      '.#####.',
      '.#####.',
      '.......',
    ]);
    const { labels, count } = labelComponents(mask, width, height, 8);
    expect(regionProperties(labels, count, width, height)[0].perimeter).toBe(16);

    // The image border counts as outside: a full 3×3 image has 8 boundary pixels.
    const full = labelComponents(new Uint8Array(9).fill(1), 3, 3, 8);
    expect(regionProperties(full.labels, full.count, 3, 3)[0].perimeter).toBe(8);

    // A single pixel is its own boundary.
    const dot = maskFrom(['...', '.#.', '...']);
    const dl = labelComponents(dot.mask, 3, 3, 8);
    expect(regionProperties(dl.labels, dl.count, 3, 3)[0].perimeter).toBe(1);
  });

  it('a large disk has circularity in [1.15, 1.35]', () => {
    // On a pixel grid the boundary-pixel count is ≈ 4√2·(r − ½) — a diagonal step counts
    // 1, not √2 — so 4πA/P² ≈ π²r² / (8(r − ½)²): about 1.26 for r = 40, tending to
    // π²/8 ≈ 1.23 for huge disks. Above the continuous value 1, but stable.
    const size = 100;
    const mask = diskMask(size, 50, 50, 40);
    const { labels, count } = labelComponents(mask, size, size, 8);
    const [disk] = regionProperties(labels, count, size, size);
    expect(disk.area).toBeGreaterThan(Math.PI * 40 * 40 * 0.98);
    expect(disk.area).toBeLessThan(Math.PI * 40 * 40 * 1.02);
    expect(disk.circularity).toBeGreaterThan(1.15);
    expect(disk.circularity).toBeLessThan(1.35);
  });

  it('ranks shapes: disk > square > thin bar', () => {
    const w = 120, h = 60;
    const mask = new Uint8Array(w * h);
    const fill = (x0: number, y0: number, x1: number, y1: number) => {
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) mask[y * w + x] = 1;
    };
    fill(2, 2, 41, 41); // 40×40 square
    fill(50, 2, 117, 5); // 68×4 bar
    const disk = diskMask(60, 30, 30, 12);
    for (let y = 0; y < 60; y++) for (let x = 0; x < 60; x++) if (disk[y * 60 + x]) mask[y * w + x + 55] = 1;
    const { labels, count } = labelComponents(mask, w, h, 8);
    const regions = regionProperties(labels, count, w, h);
    expect(count).toBe(3);
    const [square, bar, round] = regions;
    expect(square.circularity).toBeCloseTo((4 * Math.PI * 1600) / (156 * 156), 12); // P = 4·40 − 4
    expect(round.circularity).toBeGreaterThan(square.circularity);
    expect(bar.circularity).toBeLessThan(0.4);
  });

  it('square circularity only tends to π/4 for large squares (values quoted in the docs)', () => {
    // An n×n square has P = 4n − 4 boundary pixels, so 4πA/P² = πn² / (4(n − 1)²).
    const squareCircularity = (n: number) => {
      const size = n + 2;
      const mask = new Uint8Array(size * size);
      for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) mask[y * size + x] = 1;
      const { labels, count } = labelComponents(mask, size, size, 4);
      return regionProperties(labels, count, size, size)[0].circularity;
    };
    expect(squareCircularity(5)).toBeCloseTo(1.227, 3);
    expect(squareCircularity(10)).toBeCloseTo(0.970, 3);
    expect(squareCircularity(200)).toBeCloseTo(0.793, 3);
    expect(squareCircularity(200)).toBeGreaterThan(Math.PI / 4);
  });

  it('computes the mean intensity from the gray image, or null without one', () => {
    const { mask, width, height } = maskFrom(['##..', '....', '..##']);
    const gray = Float64Array.from([100, 200, 0, 0, 0, 0, 0, 0, 0, 0, 50, 60]);
    const { labels, count } = labelComponents(mask, width, height, 8);
    const [a, b] = regionProperties(labels, count, width, height, gray);
    expect(a.meanIntensity).toBe(150);
    expect(b.meanIntensity).toBe(55);
    expect(regionProperties(labels, count, width, height)[0].meanIntensity).toBeNull();
  });
});

describe('removeSmall', () => {
  const { mask, width, height } = maskFrom([
    '#.......##',
    '........##',
    '..####....',
    '..####...#',
    '..####....',
  ]);
  const { labels, count } = labelComponents(mask, width, height, 8);
  const regions = regionProperties(labels, count, width, height);

  it('drops regions under the minimum area and relabels the rest 1..count in order', () => {
    expect(regions.map(r => r.area)).toEqual([1, 4, 12, 1]);
    const out = removeSmall(labels, regions, 4);
    expect(out.count).toBe(2);
    expect(out.regions.map(r => [r.label, r.area])).toEqual([[1, 4], [2, 12]]);
    expect(new Set(out.labels)).toEqual(new Set([0, 1, 2]));
    expect(out.labels[0]).toBe(0); // the single pixel became background
    expect(out.labels[8]).toBe(1); // the 2×2 square is now region 1
    expect(out.labels[2 * width + 2]).toBe(2);
    // Measurements other than the label are unchanged.
    expect(out.regions[1].centroid).toEqual(regions[2].centroid);
  });

  it('keeps everything with minArea ≤ 1 and nothing with a huge minArea', () => {
    expect(removeSmall(labels, regions, 1).count).toBe(4);
    const none = removeSmall(labels, regions, 1000);
    expect(none.count).toBe(0);
    expect(none.labels.every(l => l === 0)).toBe(true);
  });
});

describe('labelColour', () => {
  it('is black for the background and a valid, distinct colour for each label', () => {
    expect(labelColour(0)).toEqual([0, 0, 0]);
    const seen = new Set<string>();
    for (let l = 1; l <= 50; l++) {
      const c = labelColour(l);
      for (const v of c) {
        expect(Number.isInteger(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(255);
      }
      expect(Math.max(...c)).toBeGreaterThan(120); // visible on a black background
      seen.add(c.join(','));
    }
    expect(seen.size).toBe(50);
  });

  it('gives consecutive labels clearly different hues', () => {
    for (let l = 1; l < 30; l++) {
      const [r1, g1, b1] = labelColour(l);
      const [r2, g2, b2] = labelColour(l + 1);
      expect(Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2)).toBeGreaterThan(80);
    }
  });
});
