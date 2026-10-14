import { sampleClamped } from '../processors/utils';

export interface PixelPos {
  col: number;
  row: number;
}

/** The arithmetic behind one output pixel, for the math panel. */
export interface StepMath {
  neighbourhood: number[][];
  kernel: number[][];
  products: number[][];
  sum: number;
  output: number;
}

/**
 * Visit every pixel in "snake" order: left→right on even rows, right→left on odd
 * rows, so the window never jumps across the image.
 */
export function snakeOrder(width: number, height: number): PixelPos[] {
  const order: PixelPos[] = [];
  for (let row = 0; row < height; row++) {
    if (row % 2 === 0) for (let col = 0; col < width; col++) order.push({ col, row });
    else for (let col = width - 1; col >= 0; col--) order.push({ col, row });
  }
  return order;
}

/**
 * Recompute one output pixel step by step, using the same replicate padding as
 * `convolve`, so the panel always agrees with the image.
 */
export function stepMath(
  gray: Float64Array,
  width: number,
  height: number,
  kernel: number[][],
  { col, row }: PixelPos,
  mapOutput: (sum: number) => number,
): StepMath {
  const half = (kernel.length - 1) >> 1;
  const neighbourhood: number[][] = [];
  const products: number[][] = [];
  let sum = 0;
  for (let ky = -half; ky <= half; ky++) {
    const nRow: number[] = [];
    const pRow: number[] = [];
    for (let kx = -half; kx <= half; kx++) {
      const pixel = sampleClamped(gray, width, height, col + kx, row + ky);
      const product = pixel * kernel[ky + half][kx + half];
      nRow.push(pixel);
      pRow.push(product);
      sum += product;
    }
    neighbourhood.push(nRow);
    products.push(pRow);
  }
  return { neighbourhood, kernel, products, sum, output: mapOutput(sum) };
}

/** Output image with only the first `count` pixels of `order` revealed (the rest black). */
export function revealPartial(output: Float64Array, order: PixelPos[], count: number, width: number): Float64Array {
  const partial = new Float64Array(output.length);
  for (let i = 0; i < count && i < order.length; i++) {
    const idx = order[i].row * width + order[i].col;
    partial[idx] = output[idx];
  }
  return partial;
}
