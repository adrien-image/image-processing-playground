import type { Processor } from '../types';
import { convolve, rgbaToGrayscale, grayscaleToRGBA } from '../utils';
import { SOBEL_Y } from './kernels';

/**
 * Sobel Gy processor — detects horizontal edges using only the Gy kernel.
 */
export const SobelGyProcessor: Processor = {
  id: 'sobel-gy',
  name: 'Sobel Gy — Horizontal Edges',
  description:
    'Applies the Sobel Gy 3×3 kernel, which highlights horizontal edges (top→bottom transitions). The raw convolution output is normalised to 0–255.',

  params: {},

  apply(imageData: ImageData, _config: Record<string, number | boolean | string>): ImageData {
    const { data, width, height } = imageData;
    const gray = rgbaToGrayscale(data, width, height);
    const gy = convolve(gray, width, height, SOBEL_Y);

    let maxVal = 0;
    for (let i = 0; i < width * height; i++) {
      const v = Math.abs(gy[i]);
      if (v > maxVal) maxVal = v;
    }

    const scale = maxVal > 0 ? 255 / maxVal : 1;
    const out = new Float64Array(width * height);
    for (let i = 0; i < width * height; i++) {
      out[i] = Math.abs(gy[i]) * scale;
    }

    return new ImageData(grayscaleToRGBA(out, width, height), width, height);
  },
};
