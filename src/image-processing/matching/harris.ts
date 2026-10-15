/**
 * Harris corner detector — pure TypeScript.
 * Detects corner keypoints in a grayscale image.
 */

import { SOBEL_X, SOBEL_Y } from '../processors/kernels';
import { convolve } from '../processors/utils';

export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

/**
 * Detect Harris corners in a grayscale image.
 * @param pixels - flat grayscale array (0–255), row-major
 * @param width  - image width
 * @param height - image height
 * @param maxCorners - max keypoints to return
 * @param k - Harris free parameter (typically 0.04–0.06)
 * @param quality - keep corners whose response is at least this fraction of the strongest
 * @returns array of keypoints, sorted by score descending
 */
export function detectHarris(
  pixels: Float64Array,
  width: number,
  height: number,
  maxCorners = 200,
  k = 0.05,
  quality = 0.01,
): Keypoint[] {
  // 1. Image gradients (Sobel, replicate padding)
  const gradX = convolve(pixels, width, height, SOBEL_X);
  const gradY = convolve(pixels, width, height, SOBEL_Y);

  // 2. Structure tensor summed over a 5×5 window → corner response
  //    R = det(M) − k · trace(M)²   (large positive R ⇒ corner)
  const response = new Float64Array(width * height);
  let maxResponse = 0;
  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      let ixx = 0, iyy = 0, ixy = 0;
      for (let wy = -2; wy <= 2; wy++) {
        for (let wx = -2; wx <= 2; wx++) {
          const idx = (y + wy) * width + (x + wx);
          const gx = gradX[idx];
          const gy = gradY[idx];
          ixx += gx * gx;
          iyy += gy * gy;
          ixy += gx * gy;
        }
      }
      const det = ixx * iyy - ixy * ixy;
      const trace = ixx + iyy;
      const r = det - k * trace * trace;
      response[y * width + x] = r;
      if (r > maxResponse) maxResponse = r;
    }
  }

  // 3. Non-maximum suppression in a 3×3 neighbourhood. On a plateau of equal
  //    responses, the first pixel in raster order wins, so exactly one survives.
  const minResponse = maxResponse * quality;
  const corners: Keypoint[] = [];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const val = response[idx];
      if (val <= 0 || val < minResponse) continue;

      let isMax = true;
      for (let ny = -1; ny <= 1 && isMax; ny++) {
        for (let nx = -1; nx <= 1 && isMax; nx++) {
          if (ny === 0 && nx === 0) continue;
          const nIdx = idx + ny * width + nx;
          const n = response[nIdx];
          if (n > val || (n === val && nIdx < idx)) isMax = false;
        }
      }
      if (isMax) corners.push({ x, y, score: val });
    }
  }

  corners.sort((a, b) => b.score - a.score);
  return corners.slice(0, maxCorners);
}
