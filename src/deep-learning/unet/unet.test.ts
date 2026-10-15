import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { seededRandom } from '../../lib/random';
import { makeDataset, makeSample, maskIoU } from './shapes';
import { buildUNet, predictMasks, trainEpoch, UNET_SIZE } from './unet';

describe('makeSample', () => {
  it('keeps pixels in [0, 1] and marks at least one object', () => {
    const random = seededRandom(2);
    for (let k = 0; k < 20; k++) {
      const { image, mask } = makeSample(32, random);
      image.forEach(v => { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1); });
      expect(mask.reduce((a, b) => a + b, 0)).toBeGreaterThan(10);
    }
  });

  it('defeats a single global threshold on average', () => {
    const random = seededRandom(9);
    let best = 0;
    for (const t of [0.3, 0.4, 0.5, 0.6, 0.7]) {
      let iou = 0;
      for (let k = 0; k < 30; k++) {
        const { image, mask } = makeSample(32, random);
        iou += maskIoU(Array.from(image, v => (v > t ? 1 : 0)), mask);
      }
      best = Math.max(best, iou / 30);
    }
    expect(best).toBeLessThan(0.75);
  });
});

describe('maskIoU', () => {
  it('is 1 for a perfect mask and 0 for a disjoint one', () => {
    expect(maskIoU([1, 1, 0, 0], [1, 1, 0, 0])).toBe(1);
    expect(maskIoU([0, 0, 1, 1], [1, 1, 0, 0])).toBe(0);
    expect(maskIoU([0.9, 0.2, 0.7, 0], [1, 1, 0, 0])).toBeCloseTo(1 / 3);
  });
});

describe('buildUNet', () => {
  it('maps 32×32 images to 32×32 probabilities, with or without skips, and trains', async () => {
    await tf.setBackend('cpu');
    for (const skips of [true, false]) {
      const model = buildUNet({ skips });
      expect(model.outputs[0].shape).toEqual([null, UNET_SIZE, UNET_SIZE, 1]);
      const data = makeDataset(4, UNET_SIZE, seededRandom(1));
      const loss = await trainEpoch(model, data.images, data.masks, data.count);
      expect(Number.isFinite(loss)).toBe(true);
      const p = predictMasks(model, data.images, 2);
      expect(p).toHaveLength(2 * UNET_SIZE * UNET_SIZE);
      model.dispose();
    }
  }, 60_000);
});
