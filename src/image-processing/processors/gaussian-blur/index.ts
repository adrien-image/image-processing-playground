import type { Processor } from '../types';
import { gaussianKernel } from '../kernels';

/**
 * Gaussian Blur — each output pixel is a weighted average of its neighbourhood
 * (3×3 up to 9×9), with weights from a 2D Gaussian of adjustable spread.
 */
export const GaussianBlurProcessor: Processor = {
  id: 'gaussian-blur',
  name: 'Gaussian Blur',
  description:
    'Smooths the image with a normalised Gaussian kernel. Each output pixel becomes a weighted average of its neighbours — reducing noise and softening edges. Larger σ spreads the weight more evenly; a bigger kernel lets a large σ reach further, for a stronger blur.',
  params: {
    size: { type: 'number', label: 'Kernel size', default: 3, min: 3, max: 9, step: 2 },
    sigma: { type: 'number', label: 'σ (spread)', default: 0.85, min: 0.3, max: 3, step: 0.05 },
  },
  kernel: (config) => gaussianKernel((Number(config.size) - 1) >> 1, Number(config.sigma)),
  output: 'clamp',
};
