import { describe, expect, it } from 'vitest';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { applyHomography } from '../stitching/homography';
import {
  boardCorners, boardPose, calibrate, distort, distortImage, fitDistortion, projectCorners, radialShift,
  renderCheckerboard, toNormalised, undistort, undistortImage, type Camera,
} from './calibration';

const W = 160, H = 120;
const cam: Camera = { f: 160, cx: 80, cy: 60 };
const barrel = { k1: -0.3, k2: 0.08 };

describe('distort / undistort', () => {
  it('are inverses, and leave the centre alone', () => {
    for (const p of [{ x: 0.3, y: -0.2 }, { x: -0.45, y: 0.33 }, { x: 0.05, y: 0.4 }]) {
      const q = undistort(barrel, distort(barrel, p));
      expect(q.x).toBeCloseTo(p.x, 10);
      expect(q.y).toBeCloseTo(p.y, 10);
    }
    expect(distort(barrel, { x: 0, y: 0 })).toEqual({ x: 0, y: 0 });
  });

  it('pulls points inward for barrel (k1 < 0) and outward for pincushion (k1 > 0)', () => {
    expect(radialShift(cam, { k1: -0.3, k2: 0 }, 70)).toBeLessThan(0);
    expect(radialShift(cam, { k1: 0.3, k2: 0 }, 70)).toBeGreaterThan(0);
  });
});

describe('images', () => {
  it('undistorting a distorted photo gives back the ideal one (away from the borders)', () => {
    const pose = boardPose(9, 6, 80, 60, 11, 8, 0.3);
    const ideal = renderCheckerboard(W, H, pose, 9, 6);
    const back = undistortImage(distortImage(ideal, W, H, cam, barrel), W, H, cam, barrel);
    let err = 0, n = 0;
    for (let y = 30; y < 90; y++) for (let x = 40; x < 120; x++) { err += Math.abs(back[(y * W + x) * 4] - ideal[(y * W + x) * 4]); n++; }
    expect(err / n).toBeLessThan(8); // two bilinear resamplings soften the square edges a little
  });
});

describe('calibration', () => {
  const board = boardCorners(9, 6);
  const pose = boardPose(9, 6, 80, 60, 11, 12, 0.4);

  it('fits k exactly from ideal and distorted positions', () => {
    const ideal = board.map(p => toNormalised(cam, applyHomography(pose, p)));
    const k = fitDistortion(ideal, ideal.map(p => distort(barrel, p)));
    expect(k.k1).toBeCloseTo(barrel.k1, 10);
    expect(k.k2).toBeCloseTo(barrel.k2, 10);
  });

  it('recovers the distortion and the pose from the corners alone', () => {
    const observed = projectCorners(board, pose, cam, barrel);
    const { k, H: fitted, rms } = calibrate(board, observed, cam);
    expect(k.k1).toBeCloseTo(barrel.k1, 3);
    expect(k.k2).toBeCloseTo(barrel.k2, 2);
    expect(rms[rms.length - 1]).toBeLessThan(1e-3);
    expect(rms[rms.length - 1]).toBeLessThan(rms[0]);
    const corner = applyHomography(fitted, board[0]), truth = applyHomography(pose, board[0]);
    expect(Math.hypot(corner.x - truth.x, corner.y - truth.y)).toBeLessThan(0.01);
  });

  it('degrades gracefully with noisy corner detections', () => {
    const random = seededRandom(4);
    const observed = projectCorners(board, pose, cam, barrel).map(p => ({ x: p.x + 0.3 * gaussianRandom(random), y: p.y + 0.3 * gaussianRandom(random) }));
    const { k, rms } = calibrate(board, observed, cam);
    expect(Math.abs(k.k1 - barrel.k1)).toBeLessThan(0.05);
    expect(rms[rms.length - 1]).toBeLessThan(0.5); // about the detection noise
  });
});
