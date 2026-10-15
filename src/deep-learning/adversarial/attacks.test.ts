import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { buildLeNet } from '../classification/lenet';
import { seededRandom } from '../../lib/random';
import { accuracyUnderAttack, attack } from './attacks';

const random = seededRandom(11);
const image = Float32Array.from({ length: 784 }, () => random());

const prob = (model: tf.LayersModel, x: Float32Array, c: number) =>
  tf.tidy(() => (model.predict(tf.tensor4d(x, [1, 28, 28, 1])) as tf.Tensor).dataSync()[c]);

describe('attack', () => {
  it('stays inside the ε-box and the pixel range, and lowers p(label)', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    const label = 3;
    for (const steps of [1, 5]) {
      const adv = attack(model, image, label, { epsilon: 0.1, steps });
      for (let i = 0; i < 784; i++) {
        expect(Math.abs(adv[i] - image[i])).toBeLessThanOrEqual(0.1 + 1e-6);
        expect(adv[i]).toBeGreaterThanOrEqual(0);
        expect(adv[i]).toBeLessThanOrEqual(1);
      }
      expect(prob(model, adv, label)).toBeLessThan(prob(model, image, label));
    }
    model.dispose();
  });

  it('a targeted attack raises p(target)', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    const adv = attack(model, image, 3, { epsilon: 0.05, target: 7, steps: 3 });
    expect(prob(model, adv, 7)).toBeGreaterThan(prob(model, image, 7));
    model.dispose();
  });

  it('does not leak tensors', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    attack(model, image, 1, { epsilon: 0.1 });
    const before = tf.memory().numTensors;
    attack(model, image, 1, { epsilon: 0.1, steps: 3 });
    accuracyUnderAttack(model, image, Uint8Array.from([1]), 0.1, 1);
    expect(tf.memory().numTensors).toBe(before);
    model.dispose();
  });
});
