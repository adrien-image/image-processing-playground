import { describe, expect, it } from 'vitest';
import { labelComponents, regionProperties, type Region } from './components';
import { followRegion, largestRegions, openMask, regionAt, regionPixels, strongestPixel } from './pipeline';
import { applyThreshold } from '../segmentation/otsu';

const W = 20, H = 12;

function square(mask: Uint8Array, x0: number, y0: number, size: number) {
  for (let y = y0; y < y0 + size; y++) for (let x = x0; x < x0 + size; x++) mask[y * W + x] = 1;
}

describe('openMask', () => {
  it('removes specks smaller than the kernel and keeps larger squares intact', () => {
    const mask = new Uint8Array(W * H);
    square(mask, 2, 2, 5); // 5×5 object
    square(mask, 12, 2, 2); // 2×2 speck
    mask[9 * W + 15] = 1; // single pixel
    square(mask, 14, 7, 3); // 3×3 speck

    const open3 = openMask(mask, W, H, 3);
    expect(open3[2 * W + 2]).toBe(1);
    expect(open3.reduce((a, v) => a + v, 0)).toBe(25 + 9); // 5×5 and 3×3 survive
    const open5 = openMask(mask, W, H, 5);
    expect(open5.reduce((a, v) => a + v, 0)).toBe(25); // the 3×3 speck needs a 5×5 kernel
  });
});

describe('regionAt', () => {
  const mask = new Uint8Array(W * H);
  square(mask, 2, 2, 5); // region 1: x 2..6, y 2..6
  mask[9 * W + 15] = 1; // region 2: the dot at (15, 9)
  const { labels } = labelComponents(mask, W, H, 8);

  it('returns the label of the pixel under the point', () => {
    expect(regionAt(labels, W, H, 4.3, 4.9)).toBe(1);
    expect(regionAt(labels, W, H, 15.5, 9.5, 3)).toBe(2);
  });

  it('accepts a near miss within the tolerance, measured to the pixel square', () => {
    expect(regionAt(labels, W, H, 16.8, 10.2)).toBeNull(); // no tolerance: direct hits only
    expect(regionAt(labels, W, H, 16.8, 10.2, 1.5)).toBe(2); // √(0.8² + 0.2²) ≈ 0.82 from the dot
    expect(regionAt(labels, W, H, 17.6, 9.5, 1.5)).toBeNull(); // 1.6 px away
  });

  it('picks the nearest region when two are within reach', () => {
    const m = new Uint8Array(W * H);
    m[5 * W + 5] = 1; // region 1
    m[5 * W + 9] = 1; // region 2
    const l = labelComponents(m, W, H, 8).labels;
    expect(regionAt(l, W, H, 6.9, 5.5, 3)).toBe(1); // 0.9 px from region 1, 2.1 px from region 2
    expect(regionAt(l, W, H, 8.2, 5.5, 3)).toBe(2);
  });

  it('selects nothing in empty space, even inside a region’s bounding box', () => {
    // A hollow 9×9 square (one region whose box covers its hole) and a corner gap of an L.
    const m = new Uint8Array(W * H);
    for (let y = 1; y <= 9; y++) for (let x = 1; x <= 9; x++) if (x === 1 || x === 9 || y === 1 || y === 9) m[y * W + x] = 1;
    for (let x = 12; x <= 18; x++) m[1 * W + x] = 1;
    for (let y = 1; y <= 9; y++) m[y * W + 12] = 1;
    const l = labelComponents(m, W, H, 8).labels;
    expect(regionAt(l, W, H, 5.5, 5.5, 2)).toBeNull(); // centre of the hole, 3.5 px from the frame
    expect(regionAt(l, W, H, 17.5, 8.5, 2)).toBeNull(); // inside the L's box, far from its pixels
    expect(regionAt(l, W, H, 11.5, 5.5)).toBeNull();
    expect(regionAt(l, W, H, 11.5, 5.5, 1)).toBe(2); // just left of the L's upright
  });

  it('returns null outside the image', () => {
    expect(regionAt(labels, W, H, -1, 3, 2)).toBeNull();
    expect(regionAt(labels, W, H, 3, H, 2)).toBeNull();
    expect(regionAt(labels, W, H, Number.NaN, 3, 2)).toBeNull();
  });
});

describe('strongestPixel', () => {
  /** A blob that is brightest at (cx, cy) and fades outwards, on a dark background. */
  function blob(cx: number, cy: number): Float64Array {
    return Float64Array.from({ length: W * H }, (_, i) => {
      const x = i % W, y = Math.floor(i / W);
      return Math.max(0, 250 - 30 * Math.hypot(x - cx, y - cy));
    });
  }

  it('finds the brightest pixel of the region, ties going to the first in raster order', () => {
    const values = blob(6, 5);
    const { labels, count } = labelComponents(applyThreshold(values, 100), W, H, 8);
    const [region] = regionProperties(labels, count, W, H);
    expect(strongestPixel(labels, values, W, region)).toBe(5 * W + 6);

    const flat = new Float64Array(W * H);
    const mask = new Uint8Array(W * H);
    square(mask, 3, 4, 3);
    const f = labelComponents(mask, W, H, 8);
    expect(strongestPixel(f.labels, flat, W, regionProperties(f.labels, f.count, W, H)[0])).toBe(4 * W + 3);
  });

  it('stays inside the region when the threshold rises, so a selection can follow it', () => {
    const values = blob(14, 6);
    const before = labelComponents(applyThreshold(values, 60), W, H, 8);
    const anchor = strongestPixel(before.labels, values, W, regionProperties(before.labels, before.count, W, H)[0]);
    for (const t of [100, 150, 200, 240]) {
      const after = labelComponents(applyThreshold(values, t), W, H, 8);
      expect(after.labels[anchor]).toBe(1);
    }
  });
});

describe('regionPixels / followRegion', () => {
  // A 5×5 square (region 1) with a 1-px-wide arm to the right, its brightest pixel at the arm's tip.
  const mask = new Uint8Array(W * H);
  square(mask, 2, 2, 5);
  for (let x = 7; x <= 10; x++) mask[4 * W + x] = 1;
  const values = Float64Array.from(mask, (v, i) => (v ? 100 + (i % W) : 0));
  const before = labelComponents(mask, W, H, 8);
  const region = regionProperties(before.labels, before.count, W, H)[0];
  const anchor = strongestPixel(before.labels, values, W, region);
  const pixels = regionPixels(before.labels, W, region);

  it('lists every pixel of the region in raster order', () => {
    expect(pixels.length).toBe(25 + 4);
    expect(pixels[0]).toBe(2 * W + 2);
    expect(Array.from(pixels).every((p, k) => k === 0 || p > pixels[k - 1])).toBe(true);
  });

  it('follows the anchor while it is foreground', () => {
    expect(anchor).toBe(4 * W + 10);
    expect(followRegion(before.labels, anchor, pixels)).toBe(1);
  });

  it('falls back to the label covering most of the region when an opening removes the anchor', () => {
    // A 1-px dot before the square, so the square's label changes to 2.
    const withDot = Uint8Array.from(mask);
    withDot[0] = 1;
    const opened = openMask(withDot, W, H, 3);
    expect(opened[anchor]).toBe(0); // the arm is gone
    const after = labelComponents(Uint8Array.from(opened, (v, i) => v | (i === 0 ? 1 : 0)), W, H, 8);
    expect(followRegion(after.labels, anchor, pixels)).toBe(after.labels[4 * W + 4]);
    expect(after.labels[4 * W + 4]).toBe(2);
  });

  it('returns null when none of the pixels is left', () => {
    expect(followRegion(new Int32Array(W * H), anchor, pixels)).toBeNull();
  });
});

describe('largestRegions', () => {
  const region = (label: number, area: number): Region => ({
    label, area, centroid: { x: 0, y: 0 }, bbox: { x0: 0, y0: 0, x1: 0, y1: 0 }, perimeter: 1, circularity: 1, meanIntensity: null,
  });
  const regions = [region(1, 5), region(2, 50), region(3, 1), region(4, 50), region(5, 20)];

  it('returns every region (the same array) when there are few enough', () => {
    expect(largestRegions(regions, 5)).toBe(regions);
  });

  it('keeps the largest ones, ties by label, in label order', () => {
    expect(largestRegions(regions, 3).map(r => r.label)).toEqual([2, 4, 5]);
    expect(largestRegions(regions, 1).map(r => r.label)).toEqual([2]);
    expect(largestRegions(regions, 0)).toEqual([]);
  });
});
