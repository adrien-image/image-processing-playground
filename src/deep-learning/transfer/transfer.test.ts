import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import type { MnistSplit } from '../shared/mnist';
import { buildClassifier, buildFeatureExtractor, cloneExtractor, disposeClassifier, evaluate, selectDigits, train } from './transfer';

/** Fake "digits": class d is a bright vertical bar at column 2 + 2d, plus noise. */
function fakeSplit(count: number, seed: number): MnistSplit {
  const random = seededRandom(seed);
  const labels = Uint8Array.from({ length: count }, (_, i) => i % 10);
  const images = Float32Array.from({ length: count * 784 }, (_, k) => {
    const i = Math.floor(k / 784), x = (k % 784) % 28;
    return Math.abs(x - (4 + 2 * labels[i])) <= 1 ? 0.9 : 0.1 * random();
  });
  return { images, labels, count };
}

describe('selectDigits', () => {
  it('picks the requested digits, per class, and renumbers them', () => {
    const split = fakeSplit(100, 1);
    const s = selectDigits(split, [5, 6, 7], 4, seededRandom(2));
    expect(s.count).toBe(12);
    expect([...s.labels].sort()).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
    // The image of a renumbered label is still an image of the original digit (bar at column 4 + 2·digit).
    const k = s.labels.indexOf(1);
    expect(s.images[k * 784 + 4 + 2 * 6]).toBeCloseTo(0.9, 6);
    expect(selectDigits(split, [0], Infinity, seededRandom(3)).count).toBe(10);
  });
});

describe('train', () => {
  it('pairs every image with its own label, also when the data fits in one batch', async () => {
    await tf.setBackend('cpu');
    // Two images, opposite labels: a model can only fit them if each keeps its label.
    const split = fakeSplit(20, 7);
    const data = selectDigits(split, [0, 9], 1, seededRandom(8));
    const c = buildClassifier(buildFeatureExtractor(2), 2, 0.01, false);
    const history = await train(c, data, data, 2, 40, { batchSize: 16, random: seededRandom(9) });
    expect(history[history.length - 1]).toBe(1);
    disposeClassifier(c);
  }, 60_000);
});

describe('transfer', () => {
  it('a frozen extractor keeps its weights, a clone is independent, and nothing leaks', async () => {
    await tf.setBackend('cpu');
    const split = fakeSplit(200, 4);
    const random = seededRandom(5);
    const pretrainData = selectDigits(split, [0, 1, 2, 3, 4], Infinity, random);
    const newData = selectDigits(split, [5, 6, 7, 8, 9], 4, random);
    const newTest = selectDigits(fakeSplit(100, 6), [5, 6, 7, 8, 9], Infinity, random);
    const before = tf.memory().numTensors;

    const extractor = buildFeatureExtractor(1);
    const pretrain = buildClassifier(extractor, 5, 0.01, false);
    const history = await train(pretrain, pretrainData, pretrainData, 5, 3);
    expect(history[history.length - 1]).toBeGreaterThan(0.6);

    const frozenBefore = extractor.getWeights().map(w => w.dataSync().slice());
    const probeFeatures = cloneExtractor(extractor);
    const probe = buildClassifier(probeFeatures, 5, 0.01, true);
    await train(probe, newData, newTest, 5, 3);
    probeFeatures.getWeights().forEach((w, i) => expect(Array.from(w.dataSync())).toEqual(Array.from(frozenBefore[i])));

    const copy = cloneExtractor(extractor);
    const tuned = buildClassifier(copy, 5, 0.001, false);
    await train(tuned, newData, newTest, 5, 2);
    expect(Array.from(copy.getWeights()[0].dataSync())).not.toEqual(Array.from(frozenBefore[0]));
    extractor.getWeights().forEach((w, i) => expect(Array.from(w.dataSync())).toEqual(Array.from(frozenBefore[i])));
    expect(evaluate(probe.model, newTest)).toBeGreaterThanOrEqual(0);

    disposeClassifier(tuned);
    disposeClassifier(probe);
    disposeClassifier(pretrain);
    expect(tf.memory().numTensors).toBe(before);
  }, 120_000);
});
