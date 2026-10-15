import { describe, expect, it } from 'vitest';
import { convolve } from '../../image-processing/processors/utils';
import { SOBEL_X } from '../../image-processing/processors/kernels';
import { seededRandom } from '../../lib/random';
import { descend, evaluateKernel, kernelDistance, randomKernel, zeroKernel } from './learnKernel';

const W = 16, H = 12;
const random = seededRandom(7);
const input = Float64Array.from({ length: W * H }, () => random());

describe('evaluateKernel', () => {
  it('has zero loss and zero gradient at the true kernel', () => {
    const target = convolve(input, W, H, SOBEL_X);
    const { loss, gradient } = evaluateKernel(input, target, W, H, SOBEL_X);
    expect(loss).toBeCloseTo(0, 12);
    gradient.flat().forEach(g => expect(g).toBeCloseTo(0, 12));
  });

  it('matches a finite-difference gradient', () => {
    const target = convolve(input, W, H, SOBEL_X);
    const k = randomKernel(3, seededRandom(3));
    const { gradient } = evaluateKernel(input, target, W, H, k);
    const eps = 1e-6;
    for (const [y, x] of [[0, 0], [1, 2], [2, 1]]) {
      const plus = k.map(r => [...r]);
      const minus = k.map(r => [...r]);
      plus[y][x] += eps;
      minus[y][x] -= eps;
      const numeric = (evaluateKernel(input, target, W, H, plus).loss - evaluateKernel(input, target, W, H, minus).loss) / (2 * eps);
      expect(gradient[y][x]).toBeCloseTo(numeric, 6);
    }
  });
});

describe('descend', () => {
  it('recovers a hidden kernel from input/output pairs', () => {
    const target = convolve(input, W, H, SOBEL_X);
    let state = { kernel: zeroKernel(3), velocity: zeroKernel(3) };
    for (let i = 0; i < 600; i++) state = descend(state, evaluateKernel(input, target, W, H, state.kernel).gradient, 0.2, 0.9);
    expect(kernelDistance(state.kernel, SOBEL_X)).toBeLessThan(1e-3);
  });

  it('diverges when the learning rate is far too large', () => {
    const target = convolve(input, W, H, SOBEL_X);
    let state = { kernel: zeroKernel(3), velocity: zeroKernel(3) };
    for (let i = 0; i < 40; i++) state = descend(state, evaluateKernel(input, target, W, H, state.kernel).gradient, 3);
    expect(evaluateKernel(input, target, W, H, state.kernel).loss).toBeGreaterThan(1e3);
  });
});
