import type { Processor } from '../types';
import { SOBEL_X, SOBEL_Y } from '../kernels';

/** Sobel Gx — highlights vertical edges (left↔right intensity changes). */
export const SobelXProcessor: Processor = {
  id: 'sobel-x',
  name: 'Sobel Gx — Vertical Edges',
  description:
    'Applies the Sobel Gx 3×3 kernel, which highlights vertical edges (left↔right intensity changes). The absolute response is scaled so the strongest edge is 255.',
  params: {},
  kernel: () => SOBEL_X,
  output: 'magnitude',
};

/** Sobel Gy — highlights horizontal edges (top↔bottom intensity changes). */
export const SobelYProcessor: Processor = {
  id: 'sobel-y',
  name: 'Sobel Gy — Horizontal Edges',
  description:
    'Applies the Sobel Gy 3×3 kernel, which highlights horizontal edges (top↔bottom intensity changes). The absolute response is scaled so the strongest edge is 255.',
  params: {},
  kernel: () => SOBEL_Y,
  output: 'magnitude',
};
