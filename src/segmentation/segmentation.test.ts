import { describe, expect, it } from 'vitest';
import { applyThreshold, displayHistogram, histogram256, otsuThreshold } from './otsu';
import { kmeans, labelsToImage } from './kmeans';

const levels = (counts: [number, number][]) =>
  Float64Array.from(counts.flatMap(([value, n]) => Array<number>(n).fill(value)));

describe('otsu', () => {
  it('puts the threshold halfway between two separated peaks', () => {
    const px = levels([[34, 500], [221, 300]]);
    const t = otsuThreshold(histogram256(px));
    expect(t).toBeGreaterThanOrEqual(34);
    expect(t).toBeLessThan(221);
    expect(Math.abs(t - (34 + 221) / 2)).toBeLessThanOrEqual(1);
  });

  it('applyThreshold uses the same convention as Otsu (≤ t is background)', () => {
    // With levels 10 and 11 only, Otsu must split exactly between them.
    const px = levels([[10, 50], [11, 50]]);
    const t = otsuThreshold(histogram256(px));
    expect(t).toBe(10);
    const mask = applyThreshold(px, t);
    expect(mask.slice(0, 50).every(v => v === 0)).toBe(true);
    expect(mask.slice(50).every(v => v === 1)).toBe(true);
  });

  it('has full 0–255 resolution (not quantised to display bins)', () => {
    const px = levels([[100, 100], [108, 100]]);
    expect(otsuThreshold(histogram256(px))).toBe(104);
  });

  it('displayHistogram keeps the pixel count', () => {
    const px = levels([[0, 7], [128, 5], [255, 3]]);
    const d = displayHistogram(histogram256(px));
    expect(d.reduce((a, b) => a + b, 0)).toBe(15);
    expect(d.length).toBeGreaterThanOrEqual(8);
  });
});

describe('kmeans', () => {
  it('separates three intensity bands with k=3', () => {
    const px = levels([[34, 100], [136, 100], [221, 100]]);
    const { centroids, labels } = kmeans(px, 3);
    expect(Array.from(centroids).map(Math.round)).toEqual([34, 136, 221]);
    expect(new Set(labels).size).toBe(3);
    expect(Array.from(labelsToImage(labels, centroids)).slice(0, 1)).toEqual([34]);
  });

  it('merges the two closest bands with k=2', () => {
    const px = levels([[34, 100], [136, 100], [221, 100]]);
    const { labels } = kmeans(px, 2);
    expect(labels[150]).toBe(labels[250]); // 136 and 221 end up together
    expect(labels[0]).not.toBe(labels[150]);
  });

  it('handles a flat image', () => {
    const { centroids } = kmeans(levels([[50, 20]]), 2);
    expect(Array.from(centroids)).toContain(50);
  });
});
