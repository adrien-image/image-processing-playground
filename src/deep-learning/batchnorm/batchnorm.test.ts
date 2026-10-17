import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import { accuracy, batchNormForward, buildMlp, disposeModel, percentiles, probeModel, trainBatch, unitInputs } from './batchnorm';

describe('batchNormForward', () => {
  it('gives each unit zero mean and unit variance, then applies γ and β', () => {
    const random = seededRandom(1);
    const n = 32, d = 3;
    const x = Float32Array.from({ length: n * d }, (_, i) => 5 * (i % d) + 3 * random());
    const plain = batchNormForward(x, n, d, [1, 1, 1], [0, 0, 0], 1e-8);
    for (let j = 0; j < d; j++) {
      let mean = 0, sq = 0;
      for (let i = 0; i < n; i++) { mean += plain.y[i * d + j]; sq += plain.y[i * d + j] ** 2; }
      expect(mean / n).toBeCloseTo(0, 5);
      expect(sq / n).toBeCloseTo(1, 4);
    }
    const scaled = batchNormForward(x, n, d, [2, 2, 2], [0.5, 0.5, 0.5], 1e-8);
    expect(scaled.y[4]).toBeCloseTo(2 * plain.y[4] + 0.5, 5);
  });

  it("matches TensorFlow.js's batch normalisation in training mode", async () => {
    await tf.setBackend('cpu');
    const random = seededRandom(2);
    const n = 16, d = 4;
    const x = Float32Array.from({ length: n * d }, () => 10 * random() - 3);
    const layer = tf.layers.batchNormalization({ inputShape: [d] });
    const out = tf.tidy(() => (layer.apply(tf.tensor2d(x, [n, d]), { training: true }) as tf.Tensor).dataSync());
    const ours = batchNormForward(x, n, d, [1, 1, 1, 1], [0, 0, 0, 0], 1e-3).y;
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(ours[i], 4);
    layer.dispose();
  });
});

describe('percentiles', () => {
  it('picks the sorted values at the given ranks', () => {
    expect(percentiles([5, 1, 4, 2, 3], [0, 50, 100])).toEqual([1, 3, 5]);
  });
});

describe('buildMlp', () => {
  it('builds both variants from the same weights, trains them, and frees its tensors', async () => {
    await tf.setBackend('cpu');
    const plain = buildMlp({ batchNorm: false, learningRate: 0.5 });
    const bn = buildMlp({ batchNorm: true, learningRate: 0.5 });
    // Dense biases are replaced by β in the BN variant; BN adds γ, β and two running statistics.
    expect(plain.countParams()).toBe(784 * 100 + 100 + 2 * (100 * 100 + 100) + 100 * 10 + 10);
    expect(bn.countParams()).toBe(784 * 100 + 2 * 100 * 100 + 3 * 4 * 100 + 100 * 10 + 10);
    expect(Array.from(plain.layers[0].getWeights()[0].dataSync().slice(0, 5))).toEqual(Array.from(bn.layers[0].getWeights()[0].dataSync().slice(0, 5)));

    const random = seededRandom(3);
    const count = 40;
    const labels = Uint8Array.from({ length: count }, (_, i) => i % 2);
    // Two classes: bright left half or bright right half.
    const images = Float32Array.from({ length: count * 784 }, (_, k) => {
      const i = Math.floor(k / 784), x = (k % 784) % 28;
      return (labels[i] === 0) === (x < 14) ? 0.8 + 0.2 * random() : 0.2 * random();
    });
    for (const model of [plain, bn]) {
      const first = await trainBatch(model, images, labels, count);
      let loss = first;
      for (let s = 0; s < 30; s++) loss = await trainBatch(model, images, labels, count);
      expect(loss).toBeLessThan(first);
    }
    expect(accuracy(bn, images, labels, count)).toBe(1);

    const before = tf.memory().numTensors;
    const probe = probeModel(bn);
    expect(unitInputs(probe, images, count, 3)).toHaveLength(count);
    accuracy(plain, images, labels, count);
    expect(tf.memory().numTensors).toBe(before);
    disposeModel(plain);
    disposeModel(bn);
  }, 60_000);
});
