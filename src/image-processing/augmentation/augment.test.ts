import { describe, expect, it } from 'vitest';
import { AUGMENTATIONS, sampleAugmentations, type AugmentationType } from './augment';
import { seededRandom } from '../../lib/random';

const all = Object.fromEntries(AUGMENTATIONS.map(a => [a, true])) as Record<AugmentationType, boolean>;

describe('sampleAugmentations', () => {
  it('keeps pipeline order and only enabled steps', () => {
    const steps = sampleAugmentations({ ...all, flip: false, shear: false }, seededRandom(3));
    expect(steps.map(s => s.type)).toEqual(['rotate', 'crop', 'brightness', 'contrast', 'gaussianNoise', 'saltPepper']);
  });

  it('labels show the sampled values, within their ranges', () => {
    for (let seed = 0; seed < 50; seed++) {
      const steps = sampleAugmentations(all, seededRandom(seed));
      const deg = Number(/Rotate ([+-]?\d+)°/.exec(steps[0].label)![1]);
      expect(Math.abs(deg)).toBeLessThanOrEqual(30);
      const crop = Number(/Crop (\d+)%/.exec(steps[2].label)![1]);
      expect(crop).toBeGreaterThanOrEqual(70);
      expect(crop).toBeLessThanOrEqual(90);
    }
  });

  it('is reproducible for a given seed', () => {
    const labels = (seed: number) => sampleAugmentations(all, seededRandom(seed)).map(s => s.label);
    expect(labels(7)).toEqual(labels(7));
  });
});
