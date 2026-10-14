import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { buildLeNet, LENET_PARAMS } from './lenet';

describe('buildLeNet', () => {
  it('has the parameter count the UI advertises', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    expect(model.countParams()).toBe(LENET_PARAMS);
    expect(model.outputs[0].shape).toEqual([null, 10]);
    model.dispose();
  });
});
