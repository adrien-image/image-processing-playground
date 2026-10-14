/**
 * Shared convolution kernels. Every section imports kernels from here so there is
 * exactly one definition of each.
 */

/** Sobel Gx — responds to horizontal intensity change, i.e. vertical edges. */
export const SOBEL_X: number[][] = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
];

/** Sobel Gy — responds to vertical intensity change, i.e. horizontal edges. */
export const SOBEL_Y: number[][] = [
  [-1, -2, -1],
  [ 0,  0,  0],
  [ 1,  2,  1],
];

/** Prewitt — like Sobel but with uniform (unweighted) smoothing. */
export const PREWITT_X: number[][] = [
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1],
];

export const PREWITT_Y: number[][] = [
  [-1, -1, -1],
  [ 0,  0,  0],
  [ 1,  1,  1],
];

/** 4-neighbour Laplacian (second derivative). */
export const LAPLACIAN_3x3: number[][] = [
  [ 0, -1,  0],
  [-1,  4, -1],
  [ 0, -1,  0],
];

/**
 * Square Gaussian kernel of the given radius, normalised to sum to 1.
 * radius 1 → 3×3, radius 2 → 5×5, …
 */
export function gaussianKernel(radius: number, sigma: number): number[][] {
  const size = radius * 2 + 1;
  const kernel: number[][] = [];
  let total = 0;
  for (let y = -radius; y <= radius; y++) {
    const row: number[] = [];
    for (let x = -radius; x <= radius; x++) {
      const v = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
      row.push(v);
      total += v;
    }
    kernel.push(row);
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) kernel[y][x] /= total;
  }
  return kernel;
}

/**
 * Unsharp-style sharpen kernel: the centre is boosted by `4·amount` and each
 * 4-neighbour is subtracted `amount` times. The weights always sum to 1, so flat
 * regions keep their brightness. amount = 1 gives the classic [0 −1 0; −1 5 −1; 0 −1 0].
 */
export function sharpenKernel(amount: number): number[][] {
  return [
    [0, -amount, 0],
    [-amount, 1 + 4 * amount, -amount],
    [0, -amount, 0],
  ];
}
