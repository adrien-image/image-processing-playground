/**
 * Learning a convolution kernel by gradient descent.
 *
 * We know an input image x and a target image t = K* ⋆ x made with a hidden kernel K*.
 * Starting from a guess K, we repeatedly
 *   1. predict      p = K ⋆ x                    (the same convolution as everywhere else)
 *   2. measure      L = mean((p − t)²)           (mean squared error)
 *   3. differentiate ∂L/∂K[i][j] = 2/N · Σ (p − t) · x shifted by (i, j)
 *   4. step          K ← K − η · ∂L/∂K           (η = learning rate)
 * This is exactly what a CNN does for each of its filters; here there is only one.
 */
import { convolve, sampleClamped } from '../../image-processing/processors/utils';

export type Kernel = number[][];

export interface KernelFit {
  /** K ⋆ x */
  prediction: Float64Array;
  /** mean squared error between prediction and target */
  loss: number;
  /** ∂loss/∂K, same shape as K */
  gradient: Kernel;
}

/** Prediction, loss and gradient of the loss with respect to every kernel weight. */
export function evaluateKernel(input: Float64Array, target: Float64Array, width: number, height: number, kernel: Kernel): KernelFit {
  const size = kernel.length;
  const half = (size - 1) >> 1;
  const n = width * height;
  const prediction = convolve(input, width, height, kernel);

  const residual = new Float64Array(n);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    residual[i] = prediction[i] - target[i];
    loss += residual[i] * residual[i];
  }
  loss /= n;

  // Weight K[ky][kx] multiplied the input pixel at (x + kx − half, y + ky − half) for
  // every output pixel (x, y), so its gradient sums the residual times that pixel.
  const gradient: Kernel = Array.from({ length: size }, () => new Array<number>(size).fill(0));
  for (let ky = 0; ky < size; ky++) {
    for (let kx = 0; kx < size; kx++) {
      let g = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          g += residual[y * width + x] * sampleClamped(input, width, height, x + kx - half, y + ky - half);
        }
      }
      gradient[ky][kx] = (2 * g) / n;
    }
  }
  return { prediction, loss, gradient };
}

export interface OptimizerState {
  kernel: Kernel;
  /** Running direction for momentum (zeros at the start). */
  velocity: Kernel;
}

/**
 * One gradient-descent step with optional momentum:
 *   v ← μ·v − η·∇L,   K ← K + v
 * μ = 0 is plain gradient descent.
 */
export function descend(state: OptimizerState, gradient: Kernel, learningRate: number, momentum = 0): OptimizerState {
  const velocity = state.velocity.map((row, y) => row.map((v, x) => momentum * v - learningRate * gradient[y][x]));
  const kernel = state.kernel.map((row, y) => row.map((k, x) => k + velocity[y][x]));
  return { kernel, velocity };
}

/** Euclidean distance between two kernels of the same shape. */
export function kernelDistance(a: Kernel, b: Kernel): number {
  let s = 0;
  a.forEach((row, y) => row.forEach((v, x) => { s += (v - b[y][x]) ** 2; }));
  return Math.sqrt(s);
}

export const zeroKernel = (size: number): Kernel => Array.from({ length: size }, () => new Array<number>(size).fill(0));

/** Small random weights in [−scale, scale], as networks are initialised. */
export function randomKernel(size: number, random: () => number, scale = 0.3): Kernel {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => (random() * 2 - 1) * scale));
}
