import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import type { MnistSplit } from '../shared/mnist';
import { augment, balancedSubset, buildMlp, trainRun } from './regularise';

const split = (perDigit: number): MnistSplit => {
  const count = perDigit * 10;
  const random = seededRandom(count);
  return {
    images: Float32Array.from({ length: count * 784 }, () => random()),
    labels: Uint8Array.from({ length: count }, (_, i) => Math.floor(i / perDigit)),
    count,
  };
};

describe('balancedSubset', () => {
  it('takes the same number of each digit', () => {
    const sub = balancedSubset(split(5), 30);
    expect(sub.count).toBe(30);
    for (let d = 0; d < 10; d++) expect(sub.labels.filter(l => l === d).length).toBe(3);
  });
});

describe('augment', () => {
  it('is the identity with no shift and no rotation', () => {
    const s = split(1);
    expect(augment(s.images, 2, seededRandom(1), 0, 0)).toEqual(s.images.subarray(0, 2 * 784));
  });

  it('moves a single bright pixel by a whole-pixel shift', () => {
    const img = new Float32Array(784);
    img[14 * 28 + 14] = 1;
    // random() = 1 → shift = +shift on both axes, rotation = +rotate (0 here).
    const out = augment(img, 1, () => 1, 2, 0);
    expect(out[16 * 28 + 16]).toBeCloseTo(1);
    expect(out[14 * 28 + 14]).toBeCloseTo(0);
  });
});

describe('buildMlp / trainRun', () => {
  it('builds the advertised architecture and trains without leaking tensors', async () => {
    await tf.setBackend('cpu');
    const model = buildMlp({ dropout: 0.5, l2: 1e-4 });
    expect(model.countParams()).toBe(784 * 256 + 256 + 256 * 256 + 256 + 256 * 10 + 10);
    const data = split(2);
    await trainRun(model, data, data, true, seededRandom(1), { epochs: 1 });
    const before = tf.memory().numTensors;
    const stats: number[] = [];
    await trainRun(model, data, data, true, seededRandom(1), { epochs: 1, onEpoch: s => stats.push(s.valAccuracy) });
    expect(stats).toHaveLength(1);
    expect(tf.memory().numTensors).toBe(before);
    model.dispose();
  });
});
