import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import { attentionRollout, patchify, PATCHES, TinyViT, TOKENS } from './vit';

describe('patchify', () => {
  it('puts pixel (x, y) in patch (y/7, x/7) at offset (y%7, x%7)', () => {
    const img = Float32Array.from({ length: 784 }, (_, i) => i);
    const p = patchify(img, 1);
    // pixel (x = 9, y = 15): patch row 2, column 1 → patch 9; inside: row 1, column 2
    expect(p[9 * 49 + 1 * 7 + 2]).toBe(15 * 28 + 9);
    expect(new Set(p).size).toBe(784);
  });
});

describe('attentionRollout', () => {
  it('spreads [CLS] evenly over patches when attention is uniform', () => {
    const uniform = new Float32Array(TOKENS * TOKENS).fill(1 / TOKENS);
    const r = attentionRollout([uniform, uniform], 1);
    expect(r).toHaveLength(PATCHES);
    for (let i = 1; i < PATCHES; i++) expect(r[i]).toBeCloseTo(r[0], 6);
  });
});

describe('TinyViT', () => {
  it('produces normalised attention, learns a batch, and frees its tensors', async () => {
    await tf.setBackend('cpu');
    const vit = new TinyViT();
    const random = seededRandom(1);
    const images = Float32Array.from({ length: 8 * 784 }, () => random());
    const labels = Uint8Array.from({ length: 8 }, (_, i) => i % 10);

    const { probs, attention } = vit.run(images, 2);
    expect(probs).toHaveLength(20);
    expect(attention).toHaveLength(vit.config.layers);
    for (let row = 0; row < 2 * vit.config.heads * TOKENS; row++) {
      const s = attention[0].subarray(row * TOKENS, (row + 1) * TOKENS).reduce((a, b) => a + b, 0);
      expect(s).toBeCloseTo(1, 5);
    }

    const x = tf.tensor3d(patchify(images, 8), [8, 16, 49]);
    const y = tf.oneHot(tf.tensor1d(Array.from(labels), 'int32'), 10) as tf.Tensor2D;
    const optimizer = tf.train.adam(0.01);
    const lossNow = () => tf.tidy(() => vit.loss(x, y).dataSync()[0]);
    const first = lossNow();
    for (let i = 0; i < 20; i++) optimizer.minimize(() => vit.loss(x, y), false, vit.variables);
    expect(lossNow()).toBeLessThan(first);

    const before = tf.memory().numTensors;
    vit.run(images, 2);
    vit.accuracy(images, labels, 8);
    vit.positionSimilarity();
    expect(tf.memory().numTensors).toBe(before);
    tf.dispose([x, y]);
    optimizer.dispose();
    vit.dispose();
  }, 60_000);
});
