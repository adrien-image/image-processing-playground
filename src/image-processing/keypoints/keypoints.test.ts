import { describe, expect, it } from 'vitest';
import { buildScaleSpace, detectKeypoints, gaussianBlur, repeatability, type Keypoint } from './keypoints';

/** A light Gaussian blob of standard deviation t on a dark background. */
function blob(size: number, t: number, cx = size / 2, cy = size / 2): Float64Array {
  return Float64Array.from({ length: size * size }, (_, i) => {
    const x = i % size, y = Math.floor(i / size);
    return 20 + 200 * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * t * t));
  });
}

const strongest = (kps: Keypoint[]) => kps[0];

describe('gaussianBlur', () => {
  it('keeps flat images flat and preserves the mean', () => {
    const flat = new Float64Array(20 * 15).fill(7);
    for (const v of gaussianBlur(flat, 20, 15, 2)) expect(v).toBeCloseTo(7, 10);
    const img = blob(32, 3);
    const mean = (a: Float64Array) => a.reduce((s, v) => s + v, 0) / a.length;
    expect(mean(gaussianBlur(img, 32, 32, 1.5))).toBeCloseTo(mean(img), 1);
  });
});

describe('detectKeypoints', () => {
  it('finds a blob at its centre, with σ close to its size', () => {
    for (const t of [3, 6]) {
      const kp = strongest(detectKeypoints(buildScaleSpace(blob(96, t, 50, 44), 96, 96)));
      expect(Math.hypot(kp.x - 50, kp.y - 44)).toBeLessThan(1.5);
      expect(kp.sigma / t).toBeGreaterThan(0.7);
      expect(kp.sigma / t).toBeLessThan(1.4);
      expect(kp.response).toBeLessThan(0); // light blob on dark: the Laplacian is negative at its centre
    }
  });

  it('reports twice the scale for a blob twice as big', () => {
    const small = strongest(detectKeypoints(buildScaleSpace(blob(128, 4), 128, 128)));
    const large = strongest(detectKeypoints(buildScaleSpace(blob(128, 8), 128, 128)));
    expect(large.sigma / small.sigma).toBeGreaterThan(1.7);
    expect(large.sigma / small.sigma).toBeLessThan(2.3);
  });

  it('rejects straight edges and flat regions', () => {
    const edge = Float64Array.from({ length: 64 * 64 }, (_, i) => (i % 64 < 32 ? 30 : 220));
    expect(detectKeypoints(buildScaleSpace(edge, 64, 64))).toHaveLength(0);
    expect(detectKeypoints(buildScaleSpace(new Float64Array(64 * 64).fill(90), 64, 64))).toHaveLength(0);
  });

  it('builds octaves that halve the image and double σ', () => {
    const space = buildScaleSpace(blob(64, 4), 64, 64, { octaves: 3, scalesPerOctave: 3 });
    expect(space.octaves.map(o => o.width)).toEqual([64, 32, 16]);
    expect(space.octaves[0].gaussians).toHaveLength(6);
    expect(space.octaves[0].dogs).toHaveLength(5);
    expect(space.octaves[1].sigmas[0]).toBeCloseTo(2 * space.octaves[0].sigmas[0], 10);
    expect(space.octaves[0].sigmas[3]).toBeCloseTo(2 * space.octaves[0].sigmas[0], 10);
  });
});

describe('repeatability', () => {
  it('is 1 for the same keypoints, and finds blobs again in a zoomed image', () => {
    const img = blob(96, 4, 40, 50);
    const kps = detectKeypoints(buildScaleSpace(img, 96, 96));
    expect(repeatability(kps, kps, (x, y) => [x, y], 1, 96, 96)).toBe(1);
    // The same blob, twice as large, centred where (40, 50) lands after zooming ×2 about (48, 48).
    const zoomed = detectKeypoints(buildScaleSpace(blob(96, 8, 32, 52), 96, 96));
    expect(repeatability(kps, zoomed, (x, y) => [48 + 2 * (x - 48), 48 + 2 * (y - 48)], 2, 96, 96)).toBe(1);
  });
});
