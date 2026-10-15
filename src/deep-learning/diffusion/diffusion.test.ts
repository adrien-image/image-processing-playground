import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { cosineSchedule, linearSchedule, qSample, reverseStep, sampleToy, timeEmbedding } from './diffusion';
import { buildDenoiser, sample, trainStep } from './denoiser';

describe('schedules', () => {
  it('ᾱ decreases from ≈1 to ≈0', () => {
    for (const s of [linearSchedule(1000), cosineSchedule(1000)]) {
      expect(s.alphaBars[0]).toBeGreaterThan(0.99);
      expect(s.alphaBars[999]).toBeLessThan(1e-3);
      for (let t = 1; t < 1000; t++) expect(s.alphaBars[t]).toBeLessThan(s.alphaBars[t - 1]);
    }
  });
});

describe('qSample', () => {
  it('turns any data into (nearly) unit Gaussian noise at the last step', () => {
    const s = linearSchedule(1000);
    const random = seededRandom(1);
    const n = 20_000;
    const x0 = new Float32Array(n).fill(3);
    const noise = Float32Array.from({ length: n }, () => gaussianRandom(random));
    const xt = qSample(x0, 999, s, noise);
    const mean = xt.reduce((a, b) => a + b, 0) / n;
    const variance = xt.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(variance).toBeCloseTo(1, 1);
  });
});

describe('reverseStep', () => {
  it('with the exact noise predictor, recovers a single data point from pure noise', () => {
    // If the data is one point x*, the ideal predictor is ε(x_t, t) = (x_t − √ᾱ_t·x*) / √(1 − ᾱ_t).
    const s = linearSchedule(200);
    const target = [0.7, -1.2];
    const random = seededRandom(4);
    let x: Float32Array = Float32Array.from([gaussianRandom(random), gaussianRandom(random)]);
    for (let t = 199; t >= 0; t--) {
      const a = Math.sqrt(s.alphaBars[t]), b = Math.sqrt(1 - s.alphaBars[t]);
      const eps = Float32Array.from(x, (v, i) => (v - a * target[i]) / b);
      x = reverseStep(x, eps, t, s, random);
    }
    expect(x[0]).toBeCloseTo(0.7, 1);
    expect(x[1]).toBeCloseTo(-1.2, 1);
  });
});

describe('timeEmbedding / sampleToy', () => {
  it('embeds steps as bounded, distinct vectors and samples in range', () => {
    const a = timeEmbedding(3, 100), b = timeEmbedding(50, 100);
    a.forEach(v => expect(Math.abs(v)).toBeLessThanOrEqual(1));
    expect(a).not.toEqual(b);
    for (const shape of ['spiral', 'moons', 'rings'] as const) {
      sampleToy(shape, 200, seededRandom(1)).forEach(v => expect(Math.abs(v)).toBeLessThan(2));
    }
  });
});

describe('denoiser', () => {
  it('trains on a batch and samples the right number of frames without leaking', async () => {
    await tf.setBackend('cpu');
    const model = buildDenoiser();
    const s = linearSchedule(10);
    const data = sampleToy('moons', 100, seededRandom(1));
    expect(Number.isFinite(await trainStep(model, data, s, 32, seededRandom(2)))).toBe(true);
    const before = tf.memory().numTensors;
    const frames = sample(model, s, 50, seededRandom(3));
    expect(frames).toHaveLength(11);
    expect(frames[10]).toHaveLength(100);
    expect(tf.memory().numTensors).toBe(before);
    model.dispose();
  });
});
