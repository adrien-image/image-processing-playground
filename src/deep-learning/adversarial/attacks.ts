/**
 * Adversarial examples: tiny, deliberately chosen changes to an input that flip a
 * network's prediction.
 *
 * The trick is to use the network's own gradient against it. Training asks "how should
 * the weights change to lower the loss?"; an attack asks "how should the *pixels* change
 * to raise it?" and steps every pixel by ±ε in that direction (the fast gradient sign
 * method, FGSM). Each pixel changes by at most ε, yet the effects add up across 784 pixels.
 */
import * as tf from '@tensorflow/tfjs';

const SIZE = 28;
const PIXELS = SIZE * SIZE;

/**
 * Cross-entropy loss of `labels` for a batch of images, as a function of the images.
 * The model ends in a softmax, so the loss is −log p(label).
 */
function lossOf(model: tf.LayersModel, labels: tf.Tensor2D) {
  return (images: tf.Tensor) => {
    const probs = model.apply(images, { training: false }) as tf.Tensor2D;
    return tf.neg(tf.mean(tf.log(tf.add(tf.sum(tf.mul(probs, labels), 1), 1e-7))));
  };
}

export interface AttackOptions {
  /** Largest change allowed per pixel (pixels are in [0, 1]). */
  epsilon: number;
  /** Targeted attack: push towards this class instead of away from the true one. */
  target?: number;
  /** 1 = FGSM; more = iterative attack (PGD) with step ε/4, projected back into the ε-box. */
  steps?: number;
}

/**
 * Attack one 28×28 image (784 values in [0, 1]) whose true class is `label`.
 * Returns the adversarial image, also in [0, 1] and within ε of the original.
 */
export function attack(model: tf.LayersModel, image: Float32Array, label: number, { epsilon, target, steps = 1 }: AttackOptions): Float32Array {
  return tf.tidy(() => {
    const original = tf.tensor4d(image, [1, SIZE, SIZE, 1]);
    const towards = target !== undefined;
    // Untargeted: increase the loss of the true label. Targeted: decrease the loss of the target.
    const grad = tf.grad(lossOf(model, tf.oneHot(tf.tensor1d([towards ? target : label], 'int32'), 10) as tf.Tensor2D));
    const stepSize = steps === 1 ? epsilon : epsilon / 4;
    let x: tf.Tensor = original;
    for (let k = 0; k < steps; k++) {
      const direction = tf.sign(grad(x));
      x = tf.add(x, tf.mul(direction, towards ? -stepSize : stepSize));
      // Project: stay within ε of the original and inside the valid pixel range.
      x = tf.clipByValue(tf.minimum(tf.maximum(x, tf.sub(original, epsilon)), tf.add(original, epsilon)), 0, 1);
    }
    return x.dataSync() as Float32Array;
  });
}

/**
 * Accuracy on `count` test images after an untargeted FGSM attack of strength ε
 * (ε = 0 is the clean accuracy). The whole batch is attacked in one gradient pass.
 */
export function accuracyUnderAttack(model: tf.LayersModel, images: Float32Array, labels: Uint8Array, epsilon: number, count: number): number {
  return tf.tidy(() => {
    const x = tf.tensor4d(images.subarray(0, count * PIXELS), [count, SIZE, SIZE, 1]);
    const y = tf.oneHot(tf.tensor1d(Array.from(labels.subarray(0, count)), 'int32'), 10) as tf.Tensor2D;
    // The mean loss's gradient has the same sign per pixel as each image's own loss gradient.
    const adversarial = epsilon > 0 ? tf.clipByValue(tf.add(x, tf.mul(tf.sign(tf.grad(lossOf(model, y))(x)), epsilon)), 0, 1) : x;
    const predicted = (model.predict(adversarial, { batchSize: count }) as tf.Tensor2D).argMax(1);
    return tf.mean(tf.cast(tf.equal(predicted, y.argMax(1)), 'float32')).dataSync()[0];
  });
}
