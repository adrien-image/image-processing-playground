/**
 * A tiny U-Net for 32×32 binary segmentation (~28k weights).
 *
 *   32×32×8 ──────────────────────────────── skip ──────────────┐
 *     │ pool                                                    ▼
 *   16×16×16 ─────────────── skip ─────────────┐        up → concat → 32×32×8 → 1×1 → mask
 *     │ pool                                   ▼
 *   8×8×32 (bottleneck) ─────────── up → concat → 16×16×16
 *
 * The encoder (left) shrinks the image to see context; the decoder (right) grows it
 * back to full resolution. The skip connections hand the decoder the encoder's
 * full-resolution features, so object boundaries come out sharp. Pass `skips: false`
 * to build the same network without them and compare.
 */
import * as tf from '@tensorflow/tfjs';

export const UNET_SIZE = 32;

function block(x: tf.SymbolicTensor, filters: number, name: string): tf.SymbolicTensor {
  const conv = (n: number) => tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', activation: 'relu', name: `${name}_${n}` });
  return conv(2).apply(conv(1).apply(x)) as tf.SymbolicTensor;
}

export function buildUNet({ skips = true }: { skips?: boolean } = {}): tf.LayersModel {
  const input = tf.input({ shape: [UNET_SIZE, UNET_SIZE, 1] });
  const e1 = block(input, 8, 'enc1');
  const e2 = block(tf.layers.maxPooling2d({ poolSize: 2 }).apply(e1) as tf.SymbolicTensor, 16, 'enc2');
  const bottom = block(tf.layers.maxPooling2d({ poolSize: 2 }).apply(e2) as tf.SymbolicTensor, 32, 'bottleneck');

  const up = (x: tf.SymbolicTensor, skip: tf.SymbolicTensor) => {
    const u = tf.layers.upSampling2d({ size: [2, 2] }).apply(x) as tf.SymbolicTensor;
    return skips ? (tf.layers.concatenate().apply([u, skip]) as tf.SymbolicTensor) : u;
  };
  const d2 = block(up(bottom, e2), 16, 'dec2');
  const d1 = block(up(d2, e1), 8, 'dec1');
  const output = tf.layers.conv2d({ filters: 1, kernelSize: 1, activation: 'sigmoid', name: 'mask' }).apply(d1) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: output });
  model.compile({ optimizer: tf.train.adam(0.003), loss: 'binaryCrossentropy' });
  return model;
}

/** Object probability for each pixel of `count` images. */
export function predictMasks(model: tf.LayersModel, images: Float32Array, count: number): Float32Array {
  return tf.tidy(() => (model.predict(tf.tensor4d(images.subarray(0, count * UNET_SIZE * UNET_SIZE), [count, UNET_SIZE, UNET_SIZE, 1]), { batchSize: count }) as tf.Tensor).dataSync() as Float32Array);
}

const BATCH = 32;

/** One epoch over the training set; returns the mean training loss. */
export async function trainEpoch(
  model: tf.LayersModel, images: Float32Array, masks: Float32Array, count: number,
  onBatch?: (batch: number, total: number) => void,
): Promise<number> {
  const xs = tf.tensor4d(images, [count, UNET_SIZE, UNET_SIZE, 1]);
  const ys = tf.tensor4d(masks, [count, UNET_SIZE, UNET_SIZE, 1]);
  const total = Math.ceil(count / BATCH);
  try {
    const history = await model.fit(xs, ys, {
      epochs: 1, batchSize: BATCH, shuffle: true,
      callbacks: { onBatchEnd: async (b) => { onBatch?.(b + 1, total); } },
    });
    return history.history.loss[0] as number;
  } finally {
    tf.dispose([xs, ys]);
  }
}
