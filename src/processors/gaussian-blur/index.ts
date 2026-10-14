import type { Processor } from '../types';
import { gaussianKernel } from '../kernels';

/**
 * Gaussian Blur — each output pixel is a weighted average of its 3×3 neighbourhood,
 * with weights from a 2D Gaussian of adjustable spread.
 */
export const GaussianBlurProcessor: Processor = {
  id: 'gaussian-blur',
  name: 'Gaussian Blur',
  description:
    'Smooths the image with a normalised 3×3 Gaussian kernel. Each output pixel becomes a weighted average of its neighbours — reducing noise and softening edges. Larger σ spreads the weight more evenly.',
  params: {
    sigma: { type: 'number', label: 'σ (spread)', default: 0.85, min: 0.3, max: 3, step: 0.05 },
  },
  kernel: (config) => gaussianKernel(1, Number(config.sigma)),
  output: 'clamp',
};
