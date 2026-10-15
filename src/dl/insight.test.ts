import { beforeAll, describe, expect, it, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { hotColor } from '../hough/hough';
import { buildLeNet } from './lenet';
import { activations, conv1Filters, divergingColor, occlusionColor, occlusionMap, patchPositions } from './insight';

/** A blob of "ink" so the network has something to look at. */
function digitLike(): Float32Array {
  const img = new Float32Array(784);
  for (let y = 6; y < 22; y++) for (let x = 11; x < 16; x++) img[y * 28 + x] = 1;
  return img;
}

/** A linear "classifier" whose class-0 score only counts ink in the top-left 8×8 corner. */
function cornerModel(): tf.LayersModel {
  const linear = tf.sequential({
    layers: [tf.layers.flatten({ inputShape: [28, 28, 1] }), tf.layers.dense({ units: 10, activation: 'softmax' })],
  });
  const kernel = new Float32Array(784 * 10);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) kernel[(y * 28 + x) * 10] = 0.05;
  const weights = [tf.tensor2d(kernel, [784, 10]), tf.zeros([10])];
  linear.setWeights(weights);
  tf.dispose(weights);
  return linear;
}

describe('insight', () => {
  let model: tf.LayersModel;
  beforeAll(async () => {
    await tf.setBackend('cpu');
    model = buildLeNet();
  });

  it('conv1Filters returns the six 5×5 first-layer kernels', () => {
    const filters = conv1Filters(model);
    expect(filters).toHaveLength(6);
    for (const f of filters) expect(f).toHaveLength(25);

    // Same numbers as the raw [5, 5, 1, 6] kernel
    const kernel = model.getLayer('conv1').getWeights()[0].arraySync() as number[][][][];
    expect(filters[4][2 * 5 + 3]).toBeCloseTo(kernel[2][3][0][4], 6);
    expect(filters[0][0]).toBeCloseTo(kernel[0][0][0][0], 6);
  });

  it('activations has the documented shape for each feature layer', () => {
    const maps = activations(model, digitLike());
    expect(maps.map(m => [m.name, m.height, m.width, m.channels])).toEqual([
      ['conv1', 24, 24, 6],
      ['pool1', 12, 12, 6],
      ['conv2', 8, 8, 16],
      ['pool2', 4, 4, 16],
    ]);
    for (const m of maps) {
      expect(m.data).toHaveLength(m.height * m.width * m.channels);
      for (const v of m.data) expect(v).toBeGreaterThanOrEqual(0); // after ReLU (and averages of it)
    }
  });

  it('pool1 is the 2×2 average of conv1, channel by channel', () => {
    const [conv1, pool1] = activations(model, digitLike());
    const c = 3, x = 5, y = 7;
    const at = (m: typeof conv1, cx: number, cy: number) => m.data[(c * m.height + cy) * m.width + cx];
    const mean = (at(conv1, 2 * x, 2 * y) + at(conv1, 2 * x + 1, 2 * y) + at(conv1, 2 * x, 2 * y + 1) + at(conv1, 2 * x + 1, 2 * y + 1)) / 4;
    expect(at(pool1, x, y)).toBeCloseTo(mean, 5);
  });

  it('does not leak tensors', () => {
    const input = digitLike();
    // warm-up (first calls build the feature model and initialise the backend)
    conv1Filters(model);
    activations(model, input);
    occlusionMap(model, input, 1);

    const before = tf.memory().numTensors;
    conv1Filters(model);
    activations(model, input);
    occlusionMap(model, input, 1, { patch: 4, stride: 3 });
    expect(tf.memory().numTensors).toBe(before);
  });

  it('occlusionMap is 784 finite values', () => {
    const map = occlusionMap(model, digitLike(), 7);
    expect(map).toHaveLength(784);
    expect(map.every(Number.isFinite)).toBe(true);
  });

  it('occlusionMap is 0 everywhere for an all-zero input (hiding background changes nothing)', () => {
    const map = occlusionMap(model, new Float32Array(784), 3);
    for (const v of map) expect(Math.abs(v)).toBeLessThan(1e-7);
  });

  it('occlusionMap is positive where the class relies on the pixels and 0 where it ignores them', () => {
    const linear = cornerModel();
    const input = new Float32Array(784);
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        input[y * 28 + x] = 1; // evidence in the top-left corner
        input[(y + 20) * 28 + x + 20] = 1; // irrelevant ink in the bottom-right corner
      }
    }
    const map = occlusionMap(linear, input, 0);
    expect(map[3 * 28 + 3]).toBeGreaterThan(0.1);
    expect(Math.abs(map[24 * 28 + 24])).toBeLessThan(1e-6);
    expect(Math.abs(map[14 * 28 + 27])).toBeLessThan(1e-6);
    linear.dispose();
  });

  it('occlusionMap classifies all occluded copies in one forward pass', () => {
    const linear = cornerModel();
    const dense = linear.layers[1];
    const apply = vi.spyOn(dense, 'apply');
    occlusionMap(linear, digitLike(), 0, { patch: 6, stride: 2 }); // 12 × 12 patches + the original
    expect(apply).toHaveBeenCalledTimes(1); // predict() alone would take 5 chunks of ≤ 32
    const [inputs] = apply.mock.calls[0];
    expect((Array.isArray(inputs) ? inputs[0] : inputs).shape[0]).toBe(145);
    apply.mockRestore();
    linear.dispose();
  });

  it('patchPositions covers the whole image, ending flush with the edge', () => {
    expect(patchPositions(28, 6, 2)).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]);
    expect(patchPositions(28, 5, 2).slice(-2)).toEqual([22, 23]);
    expect(patchPositions(28, 28, 2)).toEqual([0]);
    expect(() => patchPositions(28, 0, 2)).toThrow(RangeError);
  });

  it('divergingColor: blue for negative, black at zero, red for positive', () => {
    expect(divergingColor(0)).toEqual([0, 0, 0]);
    const [r1, g1, b1] = divergingColor(1);
    expect(r1).toBe(255);
    expect(r1).toBeGreaterThan(Math.max(g1, b1));
    const [r2, g2, b2] = divergingColor(-1);
    expect(b2).toBe(255);
    expect(b2).toBeGreaterThan(Math.max(r2, g2));
    expect(divergingColor(5)).toEqual(divergingColor(1)); // clamped
    expect(divergingColor(-0.5)[2]).toBeCloseTo(127.5);
    expect(divergingColor(NaN)).toEqual([0, 0, 0]);
  });

  it('occlusionColor: legend colour on background, linear in t, the digit as a light veil', () => {
    for (const view of ['heat', 'signed'] as const) {
      expect(occlusionColor(0, 0, view)).toEqual([0, 0, 0]);
      expect(occlusionColor(NaN, 0, view)).toEqual([0, 0, 0]);
      const gray = occlusionColor(0, 1, view);
      expect(gray[0]).toBeCloseTo(102);
      expect(gray[0]).toBe(gray[2]);
    }
    // On background (ink 0) the pixel is exactly the colour map — no second fade (regression).
    expect(occlusionColor(0.5, 0, 'signed')).toEqual(divergingColor(0.5));
    expect(occlusionColor(-0.3, 0, 'signed')).toEqual(divergingColor(-0.3));
    expect(occlusionColor(0.5, 0, 'signed')[0]).toBeCloseTo(occlusionColor(1, 0, 'signed')[0] / 2);
    expect(occlusionColor(0.5, 0, 'heat')).toEqual(hotColor(0.85 * 0.5));
    expect(occlusionColor(1, 0, 'heat')).toEqual(hotColor(0.85));
    expect(occlusionColor(-1, 0, 'heat')).toEqual([0, 0, 0]); // heat view shows drops only
  });
});
