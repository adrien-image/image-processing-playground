import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { Gan, MIXTURE_CENTRES, modesCovered, sampleMixture } from './gan';

describe('sampleMixture and modesCovered', () => {
  it('covers all eight modes with real data, and one mode with a collapsed generator', () => {
    const real = sampleMixture(800, seededRandom(1));
    expect(modesCovered(real, MIXTURE_CENTRES)).toBe(8);
    const random = seededRandom(2);
    const collapsed = Float32Array.from({ length: 800 }, (_, i) => MIXTURE_CENTRES[3][i % 2] + 0.05 * gaussianRandom(random));
    expect(modesCovered(collapsed, MIXTURE_CENTRES)).toBe(1);
  });
});

describe('Gan', () => {
  it('moves the generated points towards the data, and frees its tensors', async () => {
    await tf.setBackend('cpu');
    const random = seededRandom(3);
    const before = tf.memory().numTensors;
    const gan = new Gan(2e-3);
    // Target: a small blob at (1, −1).
    const target = () => Float32Array.from({ length: 128 }, (_, i) => (i % 2 ? -1 : 1) + 0.05 * gaussianRandom(random));
    const mean = (pts: Float32Array) => [0, 1].map(c => pts.filter((_, i) => i % 2 === c).reduce((a, b) => a + b, 0) / (pts.length / 2));
    const distance = (pts: Float32Array) => Math.hypot(mean(pts)[0] - 1, mean(pts)[1] + 1);
    const start = distance(gan.sample(200, random));
    let losses = { d: 0, g: 0 };
    for (let s = 0; s < 150; s++) losses = gan.step(target(), random);
    expect(Number.isFinite(losses.d) && Number.isFinite(losses.g)).toBe(true);
    expect(distance(gan.sample(200, random))).toBeLessThan(start / 2);
    const map = gan.decisionMap(8, -2, 2);
    expect(map).toHaveLength(64);
    expect(Math.min(...map)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...map)).toBeLessThanOrEqual(1);
    gan.dispose();
    expect(tf.memory().numTensors).toBe(before);
  }, 120_000);
});
