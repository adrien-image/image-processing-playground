import type { Processor } from '../types';
import { convolve, rgbaToGrayscale, grayscaleToRGBA } from '../utils';

/** Classic 3×3 sharpen kernel — emphasises edges by subtracting neighbours from 5× the centre pixel */
export const SHARPEN_3x3: number[][] = [
  [ 0, -1,  0],
  [-1,  5, -1],
  [ 0, -1,  0],
];

/**
 * Sharpen processor — enhances edges and fine detail using a 3×3 high-pass filter.
 * The kernel amplifies differences between each pixel and its neighbours.
 */
export const SharpenProcessor: Processor = {
  id: 'sharpen',
  name: 'Sharpen',
  description:
    'Enhances edges and detail using a 3×3 sharpen kernel. The centre pixel is amplified (×5) while neighbours are subtracted, making transitions more prominent.',

  params: {},

  apply(imageData: ImageData, _config: Record<string, number | boolean | string>): ImageData {
    const { data, width, height } = imageData;
    const gray = rgbaToGrayscale(data, width, height);
    const result = convolve(gray, width, height, SHARPEN_3x3);

    // Clamp to valid range
    for (let i = 0; i < width * height; i++) {
      result[i] = Math.max(0, Math.min(255, result[i]));
    }

    return new ImageData(
      grayscaleToRGBA(result, width, height),
      width,
      height,
    );
  },
};
