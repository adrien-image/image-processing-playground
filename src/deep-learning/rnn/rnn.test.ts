import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import { accuracy, buildRecurrent, disposeRecurrent, readOne, STEPS, toSequences, trainStep } from './rnn';

describe('toSequences', () => {
  it('reads rows as they are, and columns transposed', () => {
    const img = Float32Array.from({ length: 784 }, (_, i) => i);
    expect(toSequences(img, 1, 'rows')[3 * 28 + 5]).toBe(3 * 28 + 5);
    expect(toSequences(img, 1, 'columns')[3 * 28 + 5]).toBe(5 * 28 + 3); // step 3 = column 3, value 5 = row 5
  });
});

describe('recurrent network', () => {
  it('remembers an early row until the end, and frees its tensors', async () => {
    await tf.setBackend('cpu');
    // Class c: a bright row at row 2 + 3c, then only noise. The answer must be carried in h for 20+ steps.
    const random = seededRandom(1);
    const count = 30, classes = 3;
    const labels = Uint8Array.from({ length: count }, (_, i) => i % classes);
    const images = Float32Array.from({ length: count * 784 }, (_, k) => {
      const n = Math.floor(k / 784), row = Math.floor((k % 784) / 28);
      return row === 2 + 3 * labels[n] ? 1 : 0.1 * random();
    });
    const before = tf.memory().numTensors;
    const r = buildRecurrent(16, 0.01);
    const first = trainStep(r, images, labels, count);
    let loss = first;
    for (let s = 0; s < 120; s++) loss = trainStep(r, images, labels, count);
    expect(loss).toBeLessThan(first * 0.5);
    expect(accuracy(r, images, labels, count)).toBeGreaterThanOrEqual(0.9);
    const { probs, hidden } = readOne(r, images.subarray(0, 784));
    expect(probs).toHaveLength(STEPS * 10);
    expect(hidden).toHaveLength(STEPS * 16);
    for (let t = 0; t < STEPS; t++) expect(probs.subarray(t * 10, t * 10 + 10).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    disposeRecurrent(r);
    expect(tf.memory().numTensors).toBe(before);
  }, 180_000);
});
