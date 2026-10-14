import { describe, expect, it } from 'vitest';
import { confusionMatrix, selectExamples } from './evaluation';
import { MNIST_SIZE, toMnistInput } from './preprocess';
import { decodeMnist } from './mnist';

describe('selectExamples (regression: wrong predictions were never shown)', () => {
  it('returns both correct and wrong examples', () => {
    const truth = Array.from({ length: 100 }, (_, i) => i % 10);
    const pred = truth.map((t, i) => (i % 7 === 0 ? (t + 1) % 10 : t));
    const conf = truth.map((_, i) => i / 100);
    const picked = selectExamples(truth, pred, conf);
    const wrong = picked.filter(i => pred[i] !== truth[i]);
    const right = picked.filter(i => pred[i] === truth[i]);
    expect(right).toHaveLength(5);
    expect(wrong).toHaveLength(5);
    expect(new Set(right.map(i => truth[i])).size).toBe(5); // one per digit
    expect(wrong[0]).toBe(98); // most confident mistake first
  });

  it('copes with a perfect classifier', () => {
    const truth = [1, 2, 3];
    expect(selectExamples(truth, truth, [1, 1, 1])).toEqual([0, 1, 2]);
  });
});

describe('confusionMatrix', () => {
  it('counts (true, predicted) pairs', () => {
    const cm = confusionMatrix([0, 0, 1, 2], [0, 1, 1, 2], 3);
    expect(cm).toEqual([[1, 1, 0], [0, 1, 0], [0, 0, 1]]);
  });
});

describe('toMnistInput', () => {
  const W = 280;
  const draw = (x0: number, y0: number, w: number, h: number) => {
    const px = new Float32Array(W * W);
    for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) px[y * W + x] = 1;
    return px;
  };
  const stats = (img: Float32Array) => {
    let mass = 0, cx = 0, cy = 0, minX = 99, maxX = -1, minY = 99, maxY = -1;
    for (let y = 0; y < MNIST_SIZE; y++) for (let x = 0; x < MNIST_SIZE; x++) {
      const v = img[y * MNIST_SIZE + x];
      mass += v; cx += v * (x + 0.5); cy += v * (y + 0.5);
      if (v > 0.1) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
    }
    return { cx: cx / mass, cy: cy / mass, w: maxX - minX + 1, h: maxY - minY + 1 };
  };

  it('scales a small off-centre stroke to a 20 px box centred by mass', () => {
    const s = stats(toMnistInput(draw(10, 20, 15, 60), W, W)!);
    expect(s.h).toBe(20);
    expect(s.w).toBe(5);
    expect(Math.abs(s.cx - 14)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(s.cy - 14)).toBeLessThanOrEqual(0.5);
  });

  it('returns null for an empty drawing', () => {
    expect(toMnistInput(new Float32Array(W * W), W, W)).toBeNull();
  });
});

describe('decodeMnist', () => {
  it('derives labels from the split-major, digit-major layout', () => {
    const meta = { imageSize: 28, perDigit: { train: 2, val: 1, test: 1 }, splits: ['train', 'val', 'test'] as ('train' | 'val' | 'test')[] };
    const rows = 40;
    const bytes = new Uint8Array(rows * 784);
    for (let r = 0; r < rows; r++) bytes[r * 784] = r;
    const d = decodeMnist(bytes, meta);
    expect(d.train.count).toBe(20);
    expect(Array.from(d.train.labels.slice(0, 4))).toEqual([0, 0, 1, 1]);
    expect(Array.from(d.val.labels)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(d.test.images[784 * 3]).toBeCloseTo(33 / 255, 6);
  });
});
