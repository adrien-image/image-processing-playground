/**
 * Harris corner detector — pure TypeScript.
 * Detects corner keypoints in a grayscale image.
 */

import { SOBEL_X, SOBEL_Y } from '../processors/sobel/kernels';

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
 * @returns array of keypoints, sorted by score descending
 */
export function detectHarris(
  pixels: Float64Array,
  width: number,
  height: number,
  maxCorners = 200,
  k = 0.05,
): Keypoint[] {
  const gradX = new Float64Array(width * height);
  const gradY = new Float64Array(width * height);

  // Sobel 3×3 gradients
  const sobelX = SOBEL_X.flat();
  const sobelY = SOBEL_Y.flat();

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      let idx = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const p = pixels[(y + ky) * width + (x + kx)];
          gx += p * sobelX[idx];
          gy += p * sobelY[idx];
          idx++;
        }
      }
      gradX[y * width + x] = gx;
      gradY[y * width + x] = gy;
    }
  }

  // Structure tensor and corner response
  const response = new Float64Array(width * height);

  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      let ixx = 0, iyy = 0, ixy = 0;
      // Sum over 5×5 window
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
      // Harris response: det(M) - k * trace(M)^2
      const det = ixx * iyy - ixy * ixy;
      const trace = ixx + iyy;
      response[y * width + x] = det - k * trace * trace;
    }
  }

  // Non-maximum suppression (3×3 neighbourhood)
  const suppressed: Keypoint[] = [];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const val = response[y * width + x];
      if (val <= 0) continue;

      let isMax = true;
      for (let ny = -1; ny <= 1 && isMax; ny++) {
        for (let nx = -1; nx <= 1 && isMax; nx++) {
          if (ny === 0 && nx === 0) continue;
          if (response[(y + ny) * width + (x + nx)] >= val) {
            isMax = false;
          }
        }
      }

      if (isMax) {
        suppressed.push({ x, y, score: val });
      }
    }
  }

  // Sort by score descending and take top maxCorners
  suppressed.sort((a, b) => b.score - a.score);
  return suppressed.slice(0, maxCorners);
}
