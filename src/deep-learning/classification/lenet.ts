import * as tf from '@tensorflow/tfjs';
import type { MnistData, MnistSplit } from '../shared/mnist';
import { confusionMatrix, selectExamples } from './evaluation';

/** Trainable parameters of the model below (checked by a unit test). */
export const LENET_PARAMS = 44_426;

/**
 * LeNet-5 for 28×28 grayscale input:
 * conv 5×5×6 → avg-pool 2 → conv 5×5×16 → avg-pool 2 → dense 120 → dense 84 → softmax 10.
 * (The original LeNet-5 took 32×32 input, which is why it is often quoted at ~60k parameters.)
 *
 * The layers get stable names so `insight.ts` can look inside: conv1, pool1, conv2, pool2, …
 */
function leNetLayers(): tf.LayersModel {
  const input = tf.input({ shape: [28, 28, 1], name: 'input' });
  const c1 = tf.layers.conv2d({ name: 'conv1', filters: 6, kernelSize: 5, activation: 'relu', padding: 'valid' }).apply(input);
  const s2 = tf.layers.averagePooling2d({ name: 'pool1', poolSize: [2, 2], strides: [2, 2] }).apply(c1);
  const c3 = tf.layers.conv2d({ name: 'conv2', filters: 16, kernelSize: 5, activation: 'relu', padding: 'valid' }).apply(s2);
  const s4 = tf.layers.averagePooling2d({ name: 'pool2', poolSize: [2, 2], strides: [2, 2] }).apply(c3);
  const flat = tf.layers.flatten({ name: 'flatten' }).apply(s4);
  const c5 = tf.layers.dense({ name: 'fc1', units: 120, activation: 'relu' }).apply(flat);
  const f6 = tf.layers.dense({ name: 'fc2', units: 84, activation: 'relu' }).apply(c5);
  const output = tf.layers.dense({ name: 'output', units: 10, activation: 'softmax' }).apply(f6);
  return tf.model({ inputs: input, outputs: output as tf.SymbolicTensor });
}

/** A fresh LeNet-5 (random weights), ready to train with Adam. */
export function buildLeNet(): tf.LayersModel {
  const model = leNetLayers();
  model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

/**
 * Free a model's memory: its weights and, if it has one, its optimizer's state (Adam keeps
 * two running averages per weight). `model.dispose()` alone leaves an optimizer that was
 * passed to compile() as an object — as buildLeNet does, to set the learning rate.
 */
export function disposeLeNet(model: tf.LayersModel): void {
  const optimizer = model.optimizer as tf.Optimizer | undefined;
  model.dispose();
  optimizer?.dispose();
}

/**
 * A frozen copy of `model`'s current weights, to look inside while training carries on.
 * Pass the previous copy as `into` to refresh it instead of building a new one.
 * setWeights() points the copy at the current values; training then gives the model new
 * tensors, so the copy keeps the old ones. (Not compiled: it is only used to predict.)
 */
export function snapshotLeNet(model: tf.LayersModel, into?: tf.LayersModel): tf.LayersModel {
  const copy = into ?? leNetLayers();
  copy.setWeights(model.getWeights());
  return copy;
}

export interface TrainingEpochData {
  /** 1-based epoch number across all training runs of this model */
  epoch: number;
  loss: number;
  accuracy: number;
  /** Measured on the validation split — never on the test set */
  valLoss: number;
  valAccuracy: number;
}

function toTensors(split: MnistSplit) {
  return {
    xs: tf.tensor4d(split.images, [split.count, 28, 28, 1]),
    // tidy() frees the intermediate label tensor; only the one-hot result is kept.
    ys: tf.tidy(() => tf.oneHot(tf.tensor1d(split.labels, 'int32'), 10)),
  };
}

/**
 * Train for `epochs` more epochs. The validation split is evaluated after every
 * epoch; the test split is held out for `evaluateTestSet`. Stops early (and resolves)
 * when `signal` aborts.
 */
export async function trainModel(
  model: tf.LayersModel,
  data: MnistData,
  {
    epochs,
    firstEpoch = 1,
    batchSize = 128,
    onEpoch,
    onBatch,
    signal,
  }: {
    epochs: number;
    firstEpoch?: number;
    batchSize?: number;
    onEpoch?: (data: TrainingEpochData) => void;
    onBatch?: (batch: number, total: number) => void;
    signal?: AbortSignal;
  },
): Promise<void> {
  const train = toTensors(data.train);
  const val = toTensors(data.val);
  const totalBatches = Math.ceil(data.train.count / batchSize);
  try {
    await model.fit(train.xs, train.ys, {
      batchSize,
      epochs,
      shuffle: true,
      validationData: [val.xs, val.ys],
      callbacks: {
        onBatchEnd: async (batch) => {
          if (signal?.aborted) model.stopTraining = true;
          onBatch?.(batch + 1, totalBatches);
        },
        onEpochEnd: async (epoch, logs) => {
          if (signal?.aborted) {
            model.stopTraining = true;
            return;
          }
          const loss = logs?.loss ?? NaN;
          onEpoch?.({
            epoch: firstEpoch + epoch,
            loss,
            accuracy: logs?.acc ?? NaN,
            valLoss: logs?.val_loss ?? NaN,
            valAccuracy: logs?.val_acc ?? NaN,
          });
          if (Number.isNaN(loss)) model.stopTraining = true;
        },
      },
    });
  } finally {
    tf.dispose([train.xs, train.ys, val.xs, val.ys]);
  }
}

export interface TestExample {
  pixels: Float32Array;
  trueLabel: number;
  predictedLabel: number;
  confidence: number;
}

export interface EvalResult {
  confusionMatrix: number[][];
  accuracy: number;
  total: number;
  examples: TestExample[];
}

/** Evaluate on the held-out test split: confusion matrix, accuracy and examples to show. */
export function evaluateTestSet(model: tf.LayersModel, test: MnistSplit): EvalResult {
  const { predicted, confidence } = tf.tidy(() => {
    const probs = model.predict(tf.tensor4d(test.images, [test.count, 28, 28, 1])) as tf.Tensor2D;
    return { predicted: probs.argMax(1).dataSync(), confidence: probs.max(1).dataSync() };
  });

  let correct = 0;
  for (let i = 0; i < test.count; i++) if (predicted[i] === test.labels[i]) correct++;

  const examples = selectExamples(test.labels, predicted, confidence).map(i => ({
    pixels: test.images.slice(i * 784, (i + 1) * 784),
    trueLabel: test.labels[i],
    predictedLabel: predicted[i],
    confidence: confidence[i],
  }));

  return {
    confusionMatrix: confusionMatrix(test.labels, predicted),
    accuracy: correct / test.count,
    total: test.count,
    examples,
  };
}

/** Class probabilities for one 28×28 input (values 0–1). */
export function predictDigit(model: tf.LayersModel, pixels: Float32Array): Float32Array {
  return tf.tidy(() => {
    const probs = (model.predict(tf.tensor4d(pixels, [1, 28, 28, 1])) as tf.Tensor).dataSync() as Float32Array;
    return Float32Array.from(probs);
  });
}
