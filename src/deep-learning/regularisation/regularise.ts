/**
 * Overfitting and the usual cures, on MNIST.
 *
 * A two-layer perceptron (no convolutions, ~270k weights) can memorise a few hundred
 * training images perfectly while doing much worse on new ones. Three standard remedies:
 *   dropout       — randomly silence a fraction of the hidden units at every training step
 *   L2 (weight decay) — add λ·Σw² to the loss, keeping the weights small
 *   augmentation  — train on randomly shifted and rotated copies of the images
 * and the best one of all: more data.
 */
import * as tf from '@tensorflow/tfjs';
import type { MnistSplit } from '../shared/mnist';

const SIZE = 28;
const PIXELS = SIZE * SIZE;

export interface RegConfig {
  trainSize: number;
  dropout: number;
  l2: number;
  augment: boolean;
}

/** 784 → 256 → 256 → 10 with ReLU, optional dropout after each hidden layer and L2 on every kernel. */
export function buildMlp({ dropout, l2 }: Pick<RegConfig, 'dropout' | 'l2'>): tf.LayersModel {
  const regularizer = l2 > 0 ? tf.regularizers.l2({ l2 }) : undefined;
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));
  for (let i = 0; i < 2; i++) {
    model.add(tf.layers.dense({ units: 256, activation: 'relu', kernelRegularizer: regularizer }));
    if (dropout > 0) model.add(tf.layers.dropout({ rate: dropout }));
  }
  model.add(tf.layers.dense({ units: 10, activation: 'softmax', kernelRegularizer: regularizer }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return model;
}

/** The first `size / 10` images of each digit (the split is ordered digit by digit). */
export function balancedSubset(split: MnistSplit, size: number): MnistSplit {
  const per = Math.max(1, Math.floor(size / 10));
  const picked: number[] = [];
  const seen = new Array<number>(10).fill(0);
  for (let i = 0; i < split.count && picked.length < per * 10; i++) {
    const d = split.labels[i];
    if (seen[d] < per) { seen[d]++; picked.push(i); }
  }
  const images = new Float32Array(picked.length * PIXELS);
  picked.forEach((i, k) => images.set(split.images.subarray(i * PIXELS, (i + 1) * PIXELS), k * PIXELS));
  return { images, labels: Uint8Array.from(picked.map(i => split.labels[i])), count: picked.length };
}

/**
 * Randomly shift (up to ±`shift` px) and rotate (up to ±`rotate`°) each 28×28 image,
 * with bilinear sampling. Pixels that come from outside the image are 0 (background).
 */
export function augment(images: Float32Array, count: number, random: () => number, shift = 2, rotate = 12): Float32Array {
  const out = new Float32Array(count * PIXELS);
  const c = (SIZE - 1) / 2;
  for (let n = 0; n < count; n++) {
    const angle = ((random() * 2 - 1) * rotate * Math.PI) / 180;
    const dx = (random() * 2 - 1) * shift, dy = (random() * 2 - 1) * shift;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const src = images.subarray(n * PIXELS, (n + 1) * PIXELS);
    const at = (x: number, y: number) => (x < 0 || y < 0 || x >= SIZE || y >= SIZE ? 0 : src[y * SIZE + x]);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        // Inverse mapping: where in the source does output pixel (x, y) come from?
        const ux = x - c - dx, uy = y - c - dy;
        const sx = cos * ux + sin * uy + c, sy = -sin * ux + cos * uy + c;
        const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
        out[n * PIXELS + y * SIZE + x] =
          at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy) + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy;
      }
    }
  }
  return out;
}

export interface EpochStats {
  epoch: number;
  /** Measured on the clean (un-augmented) training images, with dropout off. */
  trainLoss: number;
  trainAccuracy: number;
  valLoss: number;
  valAccuracy: number;
}

const tensors = (images: Float32Array, labels: Uint8Array) => ({
  xs: tf.tensor4d(images, [labels.length, SIZE, SIZE, 1]),
  ys: tf.tidy(() => tf.oneHot(tf.tensor1d(Array.from(labels), 'int32'), 10)),
});

/** Loss and accuracy of `model` on a labelled set, in inference mode. */
function measure(model: tf.LayersModel, xs: tf.Tensor, ys: tf.Tensor): [number, number] {
  return tf.tidy(() => {
    const [loss, acc] = model.evaluate(xs, ys, { batchSize: 500 }) as tf.Scalar[];
    return [loss.dataSync()[0], acc.dataSync()[0]];
  });
}

/** Train for `epochs` epochs, reporting train and validation loss/accuracy after each one. */
export async function trainRun(
  model: tf.LayersModel,
  train: MnistSplit,
  val: MnistSplit,
  augmentImages: boolean,
  random: () => number,
  { epochs, signal, onEpoch }: { epochs: number; signal?: AbortSignal; onEpoch?: (s: EpochStats) => void },
): Promise<void> {
  const clean = tensors(train.images, train.labels);
  const v = tensors(val.images, val.labels);
  try {
    for (let epoch = 1; epoch <= epochs; epoch++) {
      if (signal?.aborted) return;
      const xs = augmentImages ? tf.tensor4d(augment(train.images, train.count, random), [train.count, SIZE, SIZE, 1]) : clean.xs;
      try {
        await model.fit(xs, clean.ys, { epochs: 1, batchSize: Math.min(64, train.count), shuffle: true });
      } finally {
        if (xs !== clean.xs) xs.dispose();
      }
      const [trainLoss, trainAccuracy] = measure(model, clean.xs, clean.ys);
      const [valLoss, valAccuracy] = measure(model, v.xs, v.ys);
      onEpoch?.({ epoch, trainLoss, trainAccuracy, valLoss, valAccuracy });
      await tf.nextFrame();
    }
  } finally {
    tf.dispose([clean.xs, clean.ys, v.xs, v.ys]);
  }
}
