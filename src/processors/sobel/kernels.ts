/** Sobel Gx kernel — detects vertical edges */
export const SOBEL_X: number[][] = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
];

/** Sobel Gy kernel — detects horizontal edges */
export const SOBEL_Y: number[][] = [
  [-1, -2, -1],
  [ 0,  0,  0],
  [ 1,  2,  1],
];
