import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import { augment, buildSimClr, disposeSimClr, embed, knnAccuracy, ntXent, pca2, REPRESENTATION, trainStep } from './contrastive';

describe('ntXent', () => {
  it('matches the formula, and is low when partners agree and others differ', async () => {
    await tf.setBackend('cpu');
    // Two images; their two views are identical, and the images are orthogonal.
    const z1 = tf.tensor2d([[1, 0], [0, 1]]), z2 = tf.tensor2d([[1, 0], [0, 1]]);
    const tau = 0.5;
    // Row 0: partner sim 1, others: view of image 1 (sim 0) twice → −log(e² / (e² + 2)).
    const expected = -Math.log(Math.exp(2) / (Math.exp(2) + 2));
    expect(ntXent(z1, z2, tau).dataSync()[0]).toBeCloseTo(expected, 5);
    // Swapping the partners makes every view's partner the wrong one: much higher loss.
    const swapped = tf.tensor2d([[0, 1], [1, 0]]);
    expect(ntXent(z1, swapped, tau).dataSync()[0]).toBeGreaterThan(expected + 1);
    tf.dispose([z1, z2, swapped]);
  });
});

describe('augment', () => {
  it('keeps the shape and range, and changes the image', () => {
    const img = new Float32Array(784);
    for (let y = 8; y < 20; y++) for (let x = 12; x < 16; x++) img[y * 28 + x] = 1;
    const out = augment(img, 1, seededRandom(1));
    expect(out).toHaveLength(784);
    expect(Math.min(...out)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...out)).toBeLessThanOrEqual(1);
    expect(Array.from(out)).not.toEqual(Array.from(img));
    const mass = (a: Float32Array) => a.reduce((s, v) => s + v, 0);
    expect(mass(augment(img, 1, seededRandom(2), { noise: 0 })) / mass(img)).toBeGreaterThan(0.6); // the stroke stays in view
  });
});

describe('knnAccuracy and pca2', () => {
  it('classifies separable clusters perfectly and finds their main axis', () => {
    const d = 3;
    const pts = (cx: number) => [cx, 1, 0.1 * cx];
    const train = Float32Array.from([...pts(5), ...pts(5.2), ...pts(-5), ...pts(-5.1)]);
    const test = Float32Array.from([...pts(4.9), ...pts(-4.8)]);
    expect(knnAccuracy(train, [0, 0, 1, 1], test, [0, 1], d, 1)).toBe(1);
    const proj = pca2(train, 4, d);
    // The first component separates the clusters by sign.
    expect(Math.sign(proj[0])).toBe(Math.sign(proj[2]));
    expect(Math.sign(proj[0])).not.toBe(Math.sign(proj[4]));
  });
});

describe('SimCLR', () => {
  it('lowers its loss on a fixed batch and frees its tensors', async () => {
    await tf.setBackend('cpu');
    const random = seededRandom(3);
    const count = 8;
    const images = Float32Array.from({ length: count * 784 }, (_, k) => {
      const i = Math.floor(k / 784), x = (k % 784) % 28, y = Math.floor((k % 784) / 28);
      return (i % 2 ? Math.abs(x - 6 - i) < 2 : Math.abs(y - 6 - i) < 2) ? 1 : 0;
    });
    const before = tf.memory().numTensors;
    const m = buildSimClr(0.003);
    const first = trainStep(m, images, count, random, 0.5);
    let loss = first;
    for (let s = 0; s < 12; s++) loss = trainStep(m, images, count, random, 0.5);
    expect(loss).toBeLessThan(first);
    expect(embed(m, images, count)).toHaveLength(count * REPRESENTATION);
    disposeSimClr(m);
    expect(tf.memory().numTensors).toBe(before);
  }, 120_000);
});
