import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import { Autoencoder, klDivergence } from './vae';

describe('klDivergence', () => {
  it('is 0 for N(0, 1) and matches the closed form otherwise', async () => {
    await tf.setBackend('cpu');
    const kl = tf.tidy(() => klDivergence(tf.tensor2d([[0, 0], [1, 2]]), tf.tensor2d([[0, 0], [Math.log(4), 0]])).arraySync());
    expect(kl[0]).toBeCloseTo(0);
    // Per dimension: ½(σ² + μ² − 1 − log σ²) → ½(4 + 1 − 1 − log 4) + ½(1 + 4 − 1 − 0)
    expect(kl[1]).toBeCloseTo(0.5 * (4 - Math.log(4)) + 2, 5);
  });
});

describe('Autoencoder', () => {
  it('trains: the loss on a fixed batch goes down, and shapes are right', async () => {
    await tf.setBackend('cpu');
    const random = seededRandom(3);
    const images = Float32Array.from({ length: 16 * 784 }, (_, i) => ((i % 784) < 392 ? 0.9 : 0.1) * random());
    const ae = new Autoencoder();
    const optimizer = tf.train.adam(0.01);
    const x = tf.tensor2d(images, [16, 784]);
    const lossNow = () => tf.tidy(() => ae.loss(x, true, 1).total.dataSync()[0]);
    const first = lossNow();
    for (let i = 0; i < 30; i++) optimizer.minimize(() => ae.loss(x, true, 1).total, false, ae.variables);
    expect(lossNow()).toBeLessThan(first);
    expect(ae.codes(images, 16)).toHaveLength(32);
    const decoded = ae.decode(Float32Array.from([0, 0, 1, -1]));
    expect(decoded).toHaveLength(2 * 784);
    decoded.forEach(v => { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1); });
    x.dispose();
    optimizer.dispose();
    ae.dispose();
  }, 60_000);
});
