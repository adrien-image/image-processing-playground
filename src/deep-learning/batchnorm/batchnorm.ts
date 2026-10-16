/**
 * Batch normalisation (Ioffe & Szegedy, 2015).
 *
 * During training each layer's inputs keep changing, because every layer before it is
 * changing too. Batch normalisation standardises each unit over the mini-batch, then
 * lets the network rescale it:
 *     x̂ = (x − μ_B) / √(σ²_B + ε),   y = γ·x̂ + β
 * μ_B and σ²_B are the batch's mean and variance per unit; γ and β are learned. At test
 * time there is no batch, so running averages of μ and σ² collected during training
 * are used instead. The result: inputs to every nonlinearity stay in a stable range, so
 * much larger learning rates work, saturating activations like the sigmoid train, and
 * the network depends less on its initialisation.
 */
import * as tf from '@tensorflow/tfjs';

export interface BatchNormResult {
  y: Float32Array;
  mean: Float32Array;
  variance: Float32Array;
}

/** The batch-norm transform of an n×d batch (row-major), written out per unit. */
export function batchNormForward(x: Float32Array, n: number, d: number, gamma: ArrayLike<number>, beta: ArrayLike<number>, eps = 1e-3): BatchNormResult {
  const mean = new Float32Array(d), variance = new Float32Array(d), y = new Float32Array(n * d);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i * d + j];
    mean[j] = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (x[i * d + j] - mean[j]) ** 2;
    variance[j] = v / n; // the biased (1/n) batch variance, as in the paper and in training mode
    const inv = 1 / Math.sqrt(variance[j] + eps);
    for (let i = 0; i < n; i++) y[i * d + j] = gamma[j] * (x[i * d + j] - mean[j]) * inv + beta[j];
  }
  return { y, mean, variance };
}

/** The p-th percentiles (0–100) of some values. */
export function percentiles(values: ArrayLike<number>, ps: number[]): number[] {
  const sorted = Float64Array.from(values).sort();
  return ps.map(p => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))))]);
}

export const HIDDEN = [100, 100, 100];

/**
 * A 784 → 100 → 100 → 100 → 10 sigmoid network (the paper's MNIST experiment). With
 * `batchNorm`, each dense layer is followed by batch normalisation before the sigmoid
 * (and drops its bias, which β replaces). The last hidden layer's pre-activation is
 * named 'probe', so its distribution can be watched. Both variants start from the same
 * weights for the same seed.
 */
export function buildMlp({ batchNorm, learningRate, seed = 1 }: { batchNorm: boolean; learningRate: number; seed?: number }): tf.LayersModel {
  const model = tf.sequential();
  HIDDEN.forEach((units, i) => {
    const last = i === HIDDEN.length - 1;
    model.add(tf.layers.dense({
      units, useBias: !batchNorm, inputShape: i === 0 ? [784] : undefined,
      kernelInitializer: tf.initializers.glorotUniform({ seed: seed + i }),
      name: last && !batchNorm ? 'probe' : undefined,
    }));
    if (batchNorm) model.add(tf.layers.batchNormalization({ name: last ? 'probe' : undefined }));
    model.add(tf.layers.activation({ activation: 'sigmoid' }));
  });
  model.add(tf.layers.dense({ units: 10, activation: 'softmax', kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 9 }) }));
  model.compile({ optimizer: tf.train.sgd(learningRate), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return model;
}

/** A model sharing `model`'s layers that outputs the probe layer (the last sigmoid's input). */
export const probeModel = (model: tf.LayersModel): tf.LayersModel =>
  tf.model({ inputs: model.inputs, outputs: model.getLayer('probe').output as tf.SymbolicTensor });

/** One SGD step on a batch; returns the loss. */
export async function trainBatch(model: tf.LayersModel, images: Float32Array, labels: ArrayLike<number>, count: number): Promise<number> {
  const x = tf.tensor2d(images, [count, 784]);
  const y = tf.oneHot(tf.tensor1d(Array.from(labels), 'int32'), 10);
  try {
    const result = await model.trainOnBatch(x, y);
    return Array.isArray(result) ? result[0] : result;
  } finally {
    tf.dispose([x, y]);
  }
}

/** Classification accuracy on `count` images. */
export function accuracy(model: tf.LayersModel, images: Float32Array, labels: ArrayLike<number>, count: number): number {
  const predicted = tf.tidy(() => (model.predict(tf.tensor2d(images, [count, 784])) as tf.Tensor).argMax(1).dataSync());
  let right = 0;
  for (let i = 0; i < count; i++) if (predicted[i] === labels[i]) right++;
  return right / count;
}

/** The probe values of one unit over a batch (inference mode: running statistics for BN). */
export function unitInputs(probe: tf.LayersModel, images: Float32Array, count: number, unit: number): Float32Array {
  return tf.tidy(() => {
    const out = probe.predict(tf.tensor2d(images, [count, 784])) as tf.Tensor2D;
    return out.slice([0, unit], [count, 1]).dataSync() as Float32Array;
  });
}

export function disposeModel(model: tf.LayersModel): void {
  model.optimizer?.dispose();
  model.dispose();
}
