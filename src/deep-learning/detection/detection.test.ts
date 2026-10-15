import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { anchors, evaluate, iou, labelAnchors, makeScene, nonMaxSuppression, simulateDetections, type Detection } from './detection';

const box = (x0: number, y0: number, x1: number, y1: number, score = 1): Detection => ({ x0, y0, x1, y1, score });

describe('iou', () => {
  it('is 1 for identical, 0 for disjoint and 1/7 for half-shifted unit squares', () => {
    expect(iou(box(0, 0, 2, 2), box(0, 0, 2, 2))).toBe(1);
    expect(iou(box(0, 0, 1, 1), box(2, 2, 3, 3))).toBe(0);
    expect(iou(box(0, 0, 2, 2), box(1, 1, 3, 3))).toBeCloseTo(1 / 7, 12);
  });
});

describe('nonMaxSuppression', () => {
  it('keeps the best of each cluster and drops low scores', () => {
    const dets = [box(0, 0, 10, 10, 0.9), box(1, 1, 11, 11, 0.8), box(50, 50, 60, 60, 0.7), box(51, 50, 61, 60, 0.95), box(90, 90, 99, 99, 0.1)];
    const steps = nonMaxSuppression(dets, 0.5, 0.2);
    expect(steps.map(s => s.kept)).toEqual([3, 0]);
    expect(steps[0].suppressed).toEqual([2]);
    expect(steps[1].suppressed).toEqual([1]);
  });

  it('keeps everything at IoU threshold 1', () => {
    const dets = [box(0, 0, 10, 10, 0.9), box(0, 0, 10, 10, 0.8)];
    expect(nonMaxSuppression(dets, 1).length).toBe(2);
  });
});

describe('evaluate', () => {
  it('gives AP 1 for perfect detections and counts duplicates as false positives', () => {
    const truth = [box(0, 0, 10, 10), box(20, 20, 30, 30)];
    const dets = [box(0, 0, 10, 10, 0.9), box(20, 20, 30, 30, 0.8), box(0, 0, 10, 10, 0.7)];
    const perfect = evaluate(dets, [0, 1], truth);
    expect(perfect.averagePrecision).toBe(1);
    const withDuplicate = evaluate(dets, [0, 1, 2], truth);
    expect(withDuplicate.falsePositives).toBe(1);
    expect(withDuplicate.averagePrecision).toBe(1); // the duplicate ranks last
  });

  it('ranks a confident false positive first → AP below 1', () => {
    const truth = [box(0, 0, 10, 10)];
    const dets = [box(50, 50, 60, 60, 0.99), box(0, 0, 10, 10, 0.5)];
    expect(evaluate(dets, [0, 1], truth).averagePrecision).toBeCloseTo(0.5, 12);
  });
});

describe('NMS on simulated detections', () => {
  it('raises average precision', () => {
    const random = seededRandom(3);
    const truth = makeScene(256, 5, random);
    const dets = simulateDetections(truth, 256, random);
    const all = evaluate(dets, dets.map((_, i) => i), truth);
    const kept = evaluate(dets, nonMaxSuppression(dets, 0.5).map(s => s.kept), truth);
    expect(kept.falsePositives).toBeLessThan(all.falsePositives);
    expect(kept.averagePrecision).toBeGreaterThanOrEqual(all.averagePrecision);
  });
});

describe('anchors', () => {
  it('makes grid² × scales × ratios anchors and gives every object a positive one', () => {
    const list = anchors(256, 8, [32, 64], [0.5, 1, 2]);
    expect(list).toHaveLength(8 * 8 * 6);
    const truth = [box(10, 10, 50, 40), box(120, 100, 220, 240)];
    const labels = labelAnchors(list, truth);
    for (let g = 0; g < truth.length; g++) {
      expect(labels.some(l => l.kind === 'positive' && l.truth === g)).toBe(true);
    }
    expect(labels.filter(l => l.kind === 'negative').length).toBeGreaterThan(300);
  });
});
