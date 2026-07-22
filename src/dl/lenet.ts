import * as tf from '@tensorflow/tfjs';

/** Build the LeNet-5 model for 28×28 grayscale input */
export function buildLeNet(): tf.LayersModel {
  const input = tf.input({ shape: [28, 28, 1] });

  const c1 = tf.layers.conv2d({ filters: 6, kernelSize: 5, activation: 'relu', padding: 'valid' }).apply(input);
  const s2 = tf.layers.averagePooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(c1);

  const c3 = tf.layers.conv2d({ filters: 16, kernelSize: 5, activation: 'relu', padding: 'valid' }).apply(s2);
  const s4 = tf.layers.averagePooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(c3);

  const flat = tf.layers.flatten().apply(s4);
  const c5 = tf.layers.dense({ units: 120, activation: 'relu' }).apply(flat);
  const f6 = tf.layers.dense({ units: 84, activation: 'relu' }).apply(c5);
  const output = tf.layers.dense({ units: 10, activation: 'softmax' }).apply(f6);

  const model = tf.model({ inputs: input, outputs: output as tf.SymbolicTensor });
  model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

export interface TrainingEpochData {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

interface MnistData {
  trainImages: Float32Array;
  trainLabels: Float32Array;
  testImages: Float32Array;
  testLabels: Float32Array;
  numTrain: number;
  numTest: number;
}

/**
 * Load MNIST data via dynamic import (code-split).
 * The heavy mnist package only loads when this function is called.
 */
export async function loadMnistData(): Promise<MnistData> {
  const mnistModule = await import('mnist');
  const mnistSet = (mnistModule.default as any)?.set ?? (mnistModule as any).set;
  const data = mnistSet(6000, 1000);
  const numTrain = data.training.length;
  const numTest = data.test.length;

  const trainImgs = new Float32Array(numTrain * 784);
  const trainLbls = new Float32Array(numTrain * 10);
  const testImgs = new Float32Array(numTest * 784);
  const testLbls = new Float32Array(numTest * 10);

  // Copy in chunks to avoid blocking the main thread on mobile
  const CHUNK = 500;
  for (let start = 0; start < numTrain; start += CHUNK) {
    const end = Math.min(start + CHUNK, numTrain);
    for (let i = start; i < end; i++) {
      trainImgs.set(data.training[i].input, i * 784);
      trainLbls.set(data.training[i].output, i * 10);
    }
    await new Promise(r => setTimeout(r, 0));
  }
  for (let start = 0; start < numTest; start += CHUNK) {
    const end = Math.min(start + CHUNK, numTest);
    for (let i = start; i < end; i++) {
      testImgs.set(data.test[i].input, i * 784);
      testLbls.set(data.test[i].output, i * 10);
    }
    await new Promise(r => setTimeout(r, 0));
  }

  return { trainImages: trainImgs, trainLabels: trainLbls, testImages: testImgs, testLabels: testLbls, numTrain, numTest };
}

/**
 * Train the LeNet model for a given number of epochs.
 */
export async function trainModel(
  model: tf.LayersModel,
  data: MnistData,
  epochs: number,
  batchSize = 128,
  onEpoch?: (data: TrainingEpochData) => void,
  onBatch?: (batch: number, total: number, logs?: tf.Logs) => void,
): Promise<void> {
  const { trainImages, trainLabels, testImages, testLabels, numTrain } = data;

  const indices: number[] = [];
  for (let i = 0; i < numTrain; i++) indices.push(i);
  tf.util.shuffle(indices);

  const shuffledImages = new Float32Array(numTrain * 784);
  const shuffledLabels = new Float32Array(numTrain * 10);
  for (let i = 0; i < numTrain; i++) {
    const idx = indices[i];
    shuffledImages.set(trainImages.subarray(idx * 784, (idx + 1) * 784), i * 784);
    shuffledLabels.set(trainLabels.subarray(idx * 10, (idx + 1) * 10), i * 10);
  }

  const xs = tf.tensor4d(shuffledImages, [numTrain, 28, 28, 1]);
  const ys = tf.tensor2d(shuffledLabels, [numTrain, 10]);
  const testXs = tf.tensor4d(testImages, [data.numTest, 28, 28, 1]);
  const testYs = tf.tensor2d(testLabels, [data.numTest, 10]);

  try {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const totalBatches = Math.ceil(numTrain / batchSize);
      const history = await model.fit(xs, ys, {
        batchSize,
        epochs: 1,
        shuffle: true,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            onBatch?.(batch + 1, totalBatches, logs);
          },
        },
      });

      const loss = (history.history.loss![0] as number) ?? 0;
      const accuracy = (history.history.acc![0] as number) ?? 0;

      // Skip per-epoch validation on mobile for speed; full eval runs at the end
      if (onEpoch) onEpoch({ epoch: epoch + 1, loss, accuracy, valLoss: 0, valAccuracy: 0 });

      if (isNaN(loss)) break;
    }
  } finally {
    xs.dispose();
    ys.dispose();
    testXs.dispose();
    testYs.dispose();
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
  examples: TestExample[];
}

/**
 * Evaluate the model on the test set.
 * Returns the confusion matrix, overall accuracy, and detailed examples.
 */
export function evaluateTestSet(model: tf.LayersModel, data: MnistData): EvalResult {
  const { testImages, testLabels, numTest } = data;

  // Build tensors
  const testXs = tf.tensor4d(testImages, [numTest, 28, 28, 1]);
  const output = model.predict(testXs) as tf.Tensor;
  const probs = output.dataSync() as Float32Array;
  output.dispose();
  testXs.dispose();

  // Initialise 10×10 confusion matrix
  const cm: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));
  let correct = 0;
  const examples: TestExample[] = [];
  const wrongIndices: number[] = [];
  const correctIndices: number[] = [];

  for (let i = 0; i < numTest; i++) {
    // Decode true label from one-hot
    let trueLabel = 0;
    for (let j = 0; j < 10; j++) {
      if (testLabels[i * 10 + j] === 1) { trueLabel = j; break; }
    }

    // Find predicted label (argmax)
    let predictedLabel = 0;
    let maxProb = 0;
    for (let j = 0; j < 10; j++) {
      const p = probs[i * 10 + j];
      if (p > maxProb) { maxProb = p; predictedLabel = j; }
    }

    cm[trueLabel][predictedLabel]++;

    if (predictedLabel === trueLabel) {
      correct++;
      correctIndices.push(i);
    } else {
      wrongIndices.push(i);
    }
  }

  const accuracy = correct / numTest;

  // Pick 10 examples: up to 5 correct (one per digit if possible) + up to 5 incorrect
  const selected = new Set<number>();

  // Pick up to 5 correct: try to show variety
  for (let digit = 0; digit < 10 && selected.size < 5; digit++) {
    const idx = correctIndices.find(i => {
      let tl = 0;
      for (let j = 0; j < 10; j++) { if (testLabels[i * 10 + j] === 1) { tl = j; break; } }
      return tl === digit && !selected.has(i);
    });
    if (idx !== undefined) selected.add(idx);
  }
  // Fill remaining with more correct if needed
  for (const idx of correctIndices) {
    if (selected.size >= 10) break;
    selected.add(idx);
  }

  // Override some with wrong predictions (up to 5)
  let wrongCount = 0;
  for (const idx of wrongIndices) {
    if (wrongCount >= 5 || selected.size >= 10) break;
    selected.add(idx);
    wrongCount++;
  }

  // Build examples
  for (const idx of selected) {
    const pixels = new Float32Array(784);
    for (let j = 0; j < 784; j++) pixels[j] = testImages[idx * 784 + j];

    let trueLabel = 0;
    for (let j = 0; j < 10; j++) { if (testLabels[idx * 10 + j] === 1) { trueLabel = j; break; } }

    let predictedLabel = 0;
    let maxProb = 0;
    for (let j = 0; j < 10; j++) {
      const p = probs[idx * 10 + j];
      if (p > maxProb) { maxProb = p; predictedLabel = j; }
    }

    examples.push({ pixels, trueLabel, predictedLabel, confidence: maxProb });
  }

  return { confusionMatrix: cm, accuracy, examples };
}

export function predictDigit(model: tf.LayersModel, imageTensor: tf.Tensor4D): Float32Array {
  const output = model.predict(imageTensor) as tf.Tensor;
  const probs = output.dataSync() as Float32Array;
  for (let i = 0; i < probs.length; i++) {
    if (isNaN(probs[i])) { probs.fill(0.1); break; }
  }
  output.dispose();
  return probs;
}
