import type { Processor } from '../types';
import { convolve, rgbaToGrayscale, grayscaleToRGBA } from '../utils';
import { SOBEL_X } from './kernels';

/**
 * Sobel Gx processor — detects vertical edges using only the Gx kernel.
 */
export const SobelProcessor: Processor = {
  id: 'sobel',
  name: 'Sobel Gx — Vertical Edges',
  description:
    'Applies the Sobel Gx 3×3 kernel, which highlights vertical edges (bright→dark transitions). The raw convolution output is normalised to 0–255.',

  params: {},

  apply(imageData: ImageData, _config: Record<string, number | boolean | string>): ImageData {
    const { data, width, height } = imageData;

    // Convert to grayscale
    const gray = rgbaToGrayscale(data, width, height);

    // Convolve with Gx only
    const gx = convolve(gray, width, height, SOBEL_X);

    // Normalise to 0–255
    let maxVal = 0;
    for (let i = 0; i < width * height; i++) {
      const v = Math.abs(gx[i]);
      if (v > maxVal) maxVal = v;
    }

    const scale = maxVal > 0 ? 255 / maxVal : 1;
    const out = new Float64Array(width * height);
    for (let i = 0; i < width * height; i++) {
      out[i] = Math.abs(gx[i]) * scale;
    }

    return new ImageData(grayscaleToRGBA(out, width, height), width, height);
  },
};
