import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { buildLeNet, disposeLeNet, LENET_PARAMS, predictDigit, snapshotLeNet, trainModel } from './lenet';
import type { MnistSplit } from './mnist';

describe('buildLeNet', () => {
  it('has the parameter count the UI advertises', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    expect(model.countParams()).toBe(LENET_PARAMS);
    expect(model.outputs[0].shape).toEqual([null, 10]);
    model.dispose();
  });

  it('names the layers that insight.ts looks at', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    expect(model.layers.map(l => l.name)).toEqual(['input', 'conv1', 'pool1', 'conv2', 'pool2', 'flatten', 'fc1', 'fc2', 'output']);
    model.dispose();
  });
});

const split = (count: number): MnistSplit => ({
  images: Float32Array.from({ length: count * 784 }, (_, i) => (i % 7) / 7),
  labels: Uint8Array.from({ length: count }, (_, i) => i % 10),
  count,
});
const tinyData = () => ({ train: split(20), val: split(10), test: split(10) });

describe('trainModel', () => {
  it('does not leak tensors from one run to the next (regression)', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    const data = tinyData();
    await trainModel(model, data, { epochs: 1, batchSize: 10 }); // creates the optimizer state
    const before = tf.memory().numTensors;
    await trainModel(model, data, { epochs: 1, batchSize: 10 });
    expect(tf.memory().numTensors).toBe(before);
    disposeLeNet(model);
  });
});

describe('disposeLeNet', () => {
  it('frees the weights and the Adam optimizer state (regression: model.dispose() kept 22 tensors)', async () => {
    await tf.setBackend('cpu');
    const before = tf.memory().numTensors;
    const model = buildLeNet();
    await trainModel(model, tinyData(), { epochs: 1, batchSize: 10 });
    disposeLeNet(model);
    expect(tf.memory().numTensors).toBe(before);
  });
});

describe('snapshotLeNet', () => {
  // Three CPU training runs: about 5 s on its own, more when the other test files run alongside.
  it('keeps the weights it was given while the model trains on', { timeout: 30_000 }, async () => {
    await tf.setBackend('cpu');
    const before = tf.memory().numTensors;
    const model = buildLeNet();
    const data = tinyData();
    const digit = data.test.images.slice(0, 784);
    await trainModel(model, data, { epochs: 1, batchSize: 10 });

    const snapshot = snapshotLeNet(model);
    const atSnapshot = predictDigit(model, digit);
    expect(Array.from(predictDigit(snapshot, digit))).toEqual(Array.from(atSnapshot));

    await trainModel(model, data, { epochs: 2, batchSize: 10 });
    expect(Array.from(predictDigit(model, digit))).not.toEqual(Array.from(atSnapshot)); // the model moved on…
    expect(Array.from(predictDigit(snapshot, digit))).toEqual(Array.from(atSnapshot)); // …the snapshot did not

    // Refreshing in place catches up without allocating anything new.
    const count = tf.memory().numTensors;
    expect(snapshotLeNet(model, snapshot)).toBe(snapshot);
    expect(Array.from(predictDigit(snapshot, digit))).toEqual(Array.from(predictDigit(model, digit)));
    expect(tf.memory().numTensors).toBe(count);

    // The two share tensors until training replaces them: disposing one leaves the other intact.
    disposeLeNet(snapshot);
    expect(predictDigit(model, digit)).toHaveLength(10);
    disposeLeNet(model);
    expect(tf.memory().numTensors).toBe(before);
  });
});
