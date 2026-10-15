import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { buildLeNet } from '../classification/lenet';
import { layerGeometry, PRESETS } from '../receptive-field/receptiveField';
import { seededRandom } from '../../lib/random';
import { CONV2_JUMP, CONV2_START, gradCam, smoothGrad, upsampleCam, vanillaGradient } from './saliency';

const random = seededRandom(5);
const image = Float32Array.from({ length: 784 }, () => random());

describe('saliency maps', () => {
  it('return 28×28 non-negative maps without leaking tensors', async () => {
    await tf.setBackend('cpu');
    const model = buildLeNet();
    gradCam(model, image, 2); // builds and caches the split models
    const before = tf.memory().numTensors;
    const maps = [vanillaGradient(model, image, 2), smoothGrad(model, image, 2, seededRandom(1), 4), gradCam(model, image, 2).map];
    for (const m of maps) {
      expect(m).toHaveLength(784);
      m.forEach(v => expect(v).toBeGreaterThanOrEqual(0));
    }
    expect(gradCam(model, image, 2).coarse).toHaveLength(64);
    expect(tf.memory().numTensors).toBe(before);
    model.dispose();
  });
});

describe('upsampleCam', () => {
  it('uses the conv2 geometry from the receptive-field calculation', () => {
    const g = layerGeometry(28, PRESETS.lenet.layers)[3]; // conv2
    // A unit centred at input coordinate c covers pixel c − 0.5 (pixels are centred at x + 0.5).
    expect(g.start + 0.5).toBe(CONV2_START);
    expect(g.jump).toBe(CONV2_JUMP);
  });

  it('reproduces each unit at its centre and interpolates between them', () => {
    const cam = Float32Array.from({ length: 4 }, (_, i) => i); // 2×2: [0 1; 2 3]
    const up = upsampleCam(cam, 2, 1.5, 2, 4); // unit centres at pixels 1 and 3
    expect(up[1 * 4 + 1]).toBeCloseTo(0);
    expect(up[1 * 4 + 3]).toBeCloseTo(1);
    expect(up[3 * 4 + 1]).toBeCloseTo(2);
    expect(up[2 * 4 + 2]).toBeCloseTo(1.5);
  });
});
