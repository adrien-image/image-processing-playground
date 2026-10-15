/** Small helpers for the lessons that train their own TensorFlow.js models. */
import * as tf from '@tensorflow/tfjs';

/**
 * Run `steps` optimisation steps of `loss(step)` (which must build its tensors inside
 * the call), yielding to the browser between steps so the page stays responsive.
 * Stops early when `signal` aborts. Returns the loss of every step that ran.
 */
export async function optimise(
  optimizer: tf.Optimizer,
  steps: number,
  loss: (step: number) => tf.Scalar,
  { signal, onStep, varList }: { signal?: AbortSignal; onStep?: (step: number, loss: number) => void; varList?: tf.Variable[] } = {},
): Promise<number[]> {
  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    if (signal?.aborted) break;
    const cost = optimizer.minimize(() => loss(s), true, varList);
    const value = cost ? (await cost.data())[0] : NaN;
    cost?.dispose();
    losses.push(value);
    onStep?.(s, value);
    if (!Number.isFinite(value)) break;
    await tf.nextFrame();
  }
  return losses;
}

/** `count` random row indices out of `n`, without replacement within the batch. */
export function batchIndices(n: number, count: number, random: () => number): number[] {
  const picked = new Set<number>();
  while (picked.size < Math.min(count, n)) picked.add(Math.floor(random() * n));
  return [...picked];
}

/** Copy rows `indices` (each `rowSize` values) out of `data`. */
export function gatherRows(data: Float32Array, rowSize: number, indices: number[]): Float32Array {
  const out = new Float32Array(indices.length * rowSize);
  indices.forEach((r, k) => out.set(data.subarray(r * rowSize, (r + 1) * rowSize), k * rowSize));
  return out;
}

/** Mean of every `window` consecutive values (for smoothing noisy per-step losses). */
export function smooth(values: readonly number[], window: number): number[] {
  const out: number[] = [];
  for (let i = 0; i + window <= values.length; i += window) {
    let s = 0;
    for (let k = i; k < i + window; k++) s += values[k];
    out.push(s / window);
  }
  return out;
}
