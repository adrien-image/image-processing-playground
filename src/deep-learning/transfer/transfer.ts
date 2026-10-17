/**
 * Transfer learning: reuse what a network learned on one task for another.
 *
 * A convolutional network trained on digits 0–4 has learned edge and stroke detectors
 * that are just as useful for 5–9. With only a handful of labelled examples of the new
 * digits, there are three choices:
 *   from scratch   a fresh network trained on the few examples: it overfits them
 *   frozen         keep the pretrained feature extractor frozen and train only a new
 *                  head: fewer parameters to fit, on features that already work
 *   fine-tune      start from the pretrained weights and train everything, with a
 *                  small learning rate so the features are adjusted, not destroyed
 * The feature extractor is LeNet's convolutional part: conv 5×5 → pool → conv 5×5 →
 * pool, whose 16 × 4 × 4 = 256 outputs are the "features". Every classifier puts the
 * same kind of new head on it: dense 84 (ReLU) → one score per class.
 */
import * as tf from '@tensorflow/tfjs';
import type { MnistSplit } from '../shared/mnist';

export const FEATURES = 256;

export function buildFeatureExtractor(seed = 1): tf.Sequential {
  const init = (s: number) => tf.initializers.glorotUniform({ seed: seed * 10 + s });
  return tf.sequential({
    name: `features_${seed}_${Math.random().toString(36).slice(2, 7)}`,
    layers: [
      tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 6, kernelSize: 5, activation: 'relu', kernelInitializer: init(1) }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({ filters: 16, kernelSize: 5, activation: 'relu', kernelInitializer: init(2) }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.flatten(),
    ],
  });
}

/** A new extractor with a copy of `source`'s weights, so training it cannot change the original. */
export function cloneExtractor(source: tf.Sequential): tf.Sequential {
  const copy = buildFeatureExtractor(99);
  copy.setWeights(source.getWeights());
  return copy;
}

export interface Classifier {
  model: tf.Sequential;
  optimizer: tf.Optimizer;
  /** The variables the optimizer updates: only the new head's when frozen. */
  variables: tf.Variable[];
}

/**
 * Extractor + a new head producing class scores (logits). Freezing is nothing more than
 * which variables the optimizer is allowed to update.
 */
export function buildClassifier(extractor: tf.Sequential, classes: number, learningRate: number, frozen: boolean, seed = 7): Classifier {
  const hidden = tf.layers.dense({ units: 84, activation: 'relu', kernelInitializer: tf.initializers.glorotUniform({ seed }) });
  const scores = tf.layers.dense({ units: classes, kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 1 }) });
  const model = tf.sequential({ layers: [extractor, hidden, scores] });
  const weights = frozen ? [...hidden.trainableWeights, ...scores.trainableWeights] : model.trainableWeights;
  return { model, optimizer: tf.train.adam(learningRate), variables: weights.map(w => w.read() as tf.Variable) };
}

export interface Subset {
  images: Float32Array;
  /** Labels renumbered 0 … digits.length − 1, in the order of `digits`. */
  labels: Uint8Array;
  count: number;
}

/** Up to `perClass` images of each digit in `digits` (all of them when perClass is Infinity), shuffled. */
export function selectDigits(split: MnistSplit, digits: number[], perClass: number, random: () => number): Subset {
  const chosen: number[] = [];
  digits.forEach(d => {
    const rows: number[] = [];
    for (let i = 0; i < split.count; i++) if (split.labels[i] === d) rows.push(i);
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    chosen.push(...rows.slice(0, Math.min(perClass, rows.length)));
  });
  for (let i = chosen.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [chosen[i], chosen[j]] = [chosen[j], chosen[i]];
  }
  const images = new Float32Array(chosen.length * 784);
  const labels = new Uint8Array(chosen.length);
  chosen.forEach((r, k) => {
    images.set(split.images.subarray(r * 784, (r + 1) * 784), k * 784);
    labels[k] = digits.indexOf(split.labels[r]);
  });
  return { images, labels, count: chosen.length };
}

/** Train for `epochs` passes in shuffled mini-batches; `onEpoch` gets the test accuracy after each. */
export async function train(
  c: Classifier, data: Subset, test: Subset, classes: number, epochs: number,
  { batchSize = 32, random = Math.random, onEpoch, signal }: {
    batchSize?: number; random?: () => number; onEpoch?: (epoch: number, testAccuracy: number) => void; signal?: AbortSignal;
  } = {},
): Promise<number[]> {
  const history: number[] = [];
  const order = Array.from({ length: data.count }, (_, i) => i);
  for (let e = 0; e < epochs && !signal?.aborted; e++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let s = 0; s < data.count; s += batchSize) {
      const rows = order.slice(s, s + batchSize);
      tf.tidy(() => {
        const x = tf.tensor4d(gather(data.images, rows), [rows.length, 28, 28, 1]);
        const y = tf.oneHot(tf.tensor1d(rows.map(r => data.labels[r]), 'int32'), classes);
        c.optimizer.minimize(() => tf.losses.softmaxCrossEntropy(y, c.model.apply(x, { training: true }) as tf.Tensor) as tf.Scalar, false, c.variables);
      });
      if (signal?.aborted) break;
      await tf.nextFrame(); // keep the page responsive between batches
    }
    const acc = evaluate(c.model, test);
    history.push(acc);
    onEpoch?.(e + 1, acc);
  }
  return history;
}

function gather(images: Float32Array, rows: number[]): Float32Array {
  const out = new Float32Array(rows.length * 784);
  rows.forEach((r, k) => out.set(images.subarray(r * 784, (r + 1) * 784), k * 784));
  return out;
}

export function evaluate(model: tf.LayersModel, test: Subset): number {
  const predicted = tf.tidy(() => (model.predict(tf.tensor4d(test.images, [test.count, 28, 28, 1])) as tf.Tensor).argMax(1).dataSync());
  let right = 0;
  for (let i = 0; i < test.count; i++) if (predicted[i] === test.labels[i]) right++;
  return right / test.count;
}

/**
 * Dispose a classifier, its optimizer and its feature extractor. Give every classifier its
 * own extractor (cloneExtractor): disposing a model disposes the layers inside it.
 */
export function disposeClassifier(c: Classifier): void {
  c.optimizer.dispose();
  c.model.dispose();
}
