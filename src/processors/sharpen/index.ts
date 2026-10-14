import type { Processor } from '../types';
import { sharpenKernel } from '../kernels';

/**
 * Sharpen — amplifies the difference between each pixel and its 4-neighbours.
 * The weights sum to 1, so flat areas keep their brightness; results outside
 * 0–255 are clamped.
 */
export const SharpenProcessor: Processor = {
  id: 'sharpen',
  name: 'Sharpen',
  description:
    'Enhances edges and detail: the centre pixel is boosted while its 4 neighbours are subtracted. Weights sum to 1, so flat regions keep their brightness and only transitions are exaggerated (then clamped to 0–255).',
  params: {
    amount: { type: 'number', label: 'Amount', default: 1, min: 0, max: 3, step: 0.25 },
  },
  kernel: (config) => sharpenKernel(Number(config.amount)),
  output: 'clamp',
};
