import type { Processor } from '../types';
import { convolve, rgbaToGrayscale, grayscaleToRGBA } from '../utils';

/** 3×3 Gaussian kernel (normalised, sums to 1) */
export const GAUSSIAN_3x3: number[][] = [
  [1 / 16, 2 / 16, 1 / 16],
  [2 / 16, 4 / 16, 2 / 16],
  [1 / 16, 2 / 16, 1 / 16],
];

/**
 * Gaussian Blur processor — smooths the image using a normalised 3×3 Gaussian kernel.
 * Each output pixel is a weighted average of its neighbours.
 */
export const GaussianBlurProcessor: Processor = {
  id: 'gaussian-blur',
  name: 'Gaussian Blur',
  description:
    'Smooths the image using a normalised 3×3 Gaussian kernel. Each output pixel becomes a weighted average of its neighbours — reducing noise and softening edges.',

  params: {},

  apply(imageData: ImageData, _config: Record<string, number | boolean | string>): ImageData {
    const { data, width, height } = imageData;
    const gray = rgbaToGrayscale(data, width, height);
    const result = convolve(gray, width, height, GAUSSIAN_3x3);

    return new ImageData(
      grayscaleToRGBA(result, width, height),
      width,
      height,
    );
  },
};
