import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { blockMatch, demoLayers, leftRightCheck, renderStereo, score } from './stereo';

const W = 120, H = 80, MAX_D = 26;

describe('renderStereo', () => {
  it('shifts each layer by its disparity between the two views', () => {
    const pair = renderStereo(W, H, demoLayers(W, H), 0, seededRandom(1));
    // A background pixel far from objects: right(x − 4) = left(x).
    const x = 70, y = 5;
    expect(pair.disparity[y * W + x]).toBe(4);
    expect(pair.right[y * W + x - 4]).toBeCloseTo(pair.left[y * W + x], 9);
    expect(pair.occluded.some(v => v === 1)).toBe(true);
  });
});

describe('blockMatch', () => {
  const pair = renderStereo(W, H, demoLayers(W, H), 3, seededRandom(2));
  const run = (radius: number) => blockMatch(pair.left, pair.right, W, H, MAX_D, radius);

  it('recovers most disparities with a medium window, better than with a 1×1 window', () => {
    const tiny = score(run(0), pair, MAX_D).bad;
    const medium = score(run(3), pair, MAX_D).bad;
    expect(medium).toBeLessThan(0.1);
    expect(tiny).toBeGreaterThan(medium * 2);
  });

  it('the left–right check rejects occluded pixels', () => {
    const left = run(3);
    const right = blockMatch(pair.left, pair.right, W, H, MAX_D, 3, true);
    const checked = leftRightCheck(left, right, W, H);
    expect(score(checked, pair, MAX_D).occlusionsCaught).toBeGreaterThan(0.6);
    expect(score(checked, pair, MAX_D).bad).toBeLessThanOrEqual(score(left, pair, MAX_D).bad);
  });
});
