/**
 * A recurrent network that reads a digit one row at a time.
 *
 * Instead of seeing the whole image, a GRU (gated recurrent unit) receives the 28 rows
 * as a sequence of 28 vectors of 28 pixels. After each row it updates a hidden state h:
 *     z = σ(W_z·x + U_z·h)              update gate: how much of h to replace
 *     r = σ(W_r·x + U_r·h)              reset gate: how much of h the candidate uses
 *     ĥ = tanh(W·x + U·(r ⊙ h))         candidate state
 *     h ← (1 − z) ⊙ h + z ⊙ ĥ
 * (biases omitted). The same weights are used at every row, so the network has to
 * carry everything it needs in h. A linear layer reads a prediction from h after every
 * row; training only scores the last one, but the earlier ones show how the network's
 * opinion forms as more of the digit arrives. Reading by columns instead of rows is the
 * same network on a different sequence.
 */
import * as tf from '@tensorflow/tfjs';

export const STEPS = 28;
export type ReadOrder = 'rows' | 'columns';

/** Images (n × 784) as n sequences of 28 steps × 28 values, by rows or by columns. */
export function toSequences(images: Float32Array, count: number, order: ReadOrder): Float32Array {
  if (order === 'rows') return images.slice(0, count * 784);
  const out = new Float32Array(count * 784);
  for (let n = 0; n < count; n++) for (let t = 0; t < 28; t++) for (let k = 0; k < 28; k++) out[n * 784 + t * 28 + k] = images[n * 784 + k * 28 + t];
  return out;
}

export interface Recurrent {
  model: tf.Sequential;
  /** The same network, outputting the hidden state after every step. */
  hidden: tf.LayersModel;
  optimizer: tf.Optimizer;
  variables: tf.Variable[];
  units: number;
}

export function buildRecurrent(units = 64, learningRate = 0.003, seed = 1): Recurrent {
  const gru = tf.layers.gru({
    units, returnSequences: true, inputShape: [STEPS, 28],
    kernelInitializer: tf.initializers.glorotUniform({ seed }),
    recurrentInitializer: tf.initializers.orthogonal({ seed: seed + 1 }),
  });
  const readout = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: 10, kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 2 }) }) });
  const model = tf.sequential({ layers: [gru, readout] });
  const hidden = tf.model({ inputs: model.inputs, outputs: gru.output as tf.SymbolicTensor });
  return { model, hidden, optimizer: tf.train.adam(learningRate), variables: model.trainableWeights.map(w => w.read() as tf.Variable), units };
}

/** One step on a batch: cross-entropy of the prediction after the last row. Returns the loss. */
export function trainStep(r: Recurrent, sequences: Float32Array, labels: ArrayLike<number>, count: number): number {
  const cost = tf.tidy(() => {
    const x = tf.tensor3d(sequences, [count, STEPS, 28]);
    const y = tf.oneHot(tf.tensor1d(Array.from(labels), 'int32'), 10);
    return r.optimizer.minimize(() => {
      const logits = r.model.apply(x, { training: true }) as tf.Tensor3D;
      const last = logits.slice([0, STEPS - 1, 0], [count, 1, 10]).reshape([count, 10]);
      return tf.losses.softmaxCrossEntropy(y, last) as tf.Scalar;
    }, true, r.variables) ?? tf.scalar(NaN);
  });
  const value = cost.dataSync()[0];
  cost.dispose();
  return value;
}

/** Class probabilities after each of the 28 steps (28 × 10), and the hidden states (28 × units). */
export function readOne(r: Recurrent, sequence: Float32Array): { probs: Float32Array; hidden: Float32Array } {
  return tf.tidy(() => {
    const x = tf.tensor3d(sequence, [1, STEPS, 28]);
    return {
      probs: (r.model.predict(x) as tf.Tensor).softmax().dataSync() as Float32Array,
      hidden: (r.hidden.predict(x) as tf.Tensor).dataSync() as Float32Array,
    };
  });
}

/** Accuracy of the prediction after step `step` (1–28, default the last). */
export function accuracy(r: Recurrent, sequences: Float32Array, labels: ArrayLike<number>, count: number, step = STEPS): number {
  const predicted = tf.tidy(() => {
    const logits = r.model.predict(tf.tensor3d(sequences, [count, STEPS, 28])) as tf.Tensor3D;
    return logits.slice([0, step - 1, 0], [count, 1, 10]).reshape([count, 10]).argMax(1).dataSync();
  });
  let right = 0;
  for (let i = 0; i < count; i++) if (predicted[i] === labels[i]) right++;
  return right / count;
}

export function disposeRecurrent(r: Recurrent): void {
  r.optimizer.dispose();
  r.model.dispose();
}
