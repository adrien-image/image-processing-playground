/**
 * The learned half of diffusion: a small MLP ε_θ(x_t, t) for 2-D points that predicts
 * the noise in x_t. Input = (x, y) plus a 16-number embedding of t; output = (ε_x, ε_y).
 * Training: pick data points x_0, random steps t and noise ε, form x_t (diffusion.ts),
 * and minimise ‖ε − ε_θ(x_t, t)‖². Sampling: start from noise and apply reverseStep T times.
 */
import * as tf from '@tensorflow/tfjs';
import { gaussianRandom } from '../../lib/random';
import { qSample, reverseStep, timeEmbedding, type Schedule } from './diffusion';

const EMBED = 16;

export function buildDenoiser(): tf.Sequential {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [2 + EMBED], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 2 }));
  model.compile({ optimizer: tf.train.adam(0.002), loss: 'meanSquaredError' });
  return model;
}

/** Network inputs for points `x` (pairs) all at step `t`, or one step per point. */
function inputs(x: Float32Array, t: number | Int32Array, steps: number): Float32Array {
  const n = x.length / 2;
  const out = new Float32Array(n * (2 + EMBED));
  const shared = typeof t === 'number' ? timeEmbedding(t, steps, EMBED) : null;
  for (let i = 0; i < n; i++) {
    out[i * (2 + EMBED)] = x[2 * i];
    out[i * (2 + EMBED) + 1] = x[2 * i + 1];
    out.set(shared ?? timeEmbedding((t as Int32Array)[i], steps, EMBED), i * (2 + EMBED) + 2);
  }
  return out;
}

/** One training step on a batch of data points; returns the loss. */
export async function trainStep(model: tf.LayersModel, data: Float32Array, schedule: Schedule, batch: number, random: () => number): Promise<number> {
  const steps = schedule.betas.length;
  const n = data.length / 2;
  const x0 = new Float32Array(batch * 2);
  const noise = new Float32Array(batch * 2);
  const t = new Int32Array(batch);
  const xt = new Float32Array(batch * 2);
  for (let i = 0; i < batch; i++) {
    const r = Math.floor(random() * n);
    x0[2 * i] = data[2 * r];
    x0[2 * i + 1] = data[2 * r + 1];
    t[i] = Math.floor(random() * steps);
    noise[2 * i] = gaussianRandom(random);
    noise[2 * i + 1] = gaussianRandom(random);
    xt.set(qSample(x0.subarray(2 * i, 2 * i + 2), t[i], schedule, noise.subarray(2 * i, 2 * i + 2)), 2 * i);
  }
  const xs = tf.tensor2d(inputs(xt, t, steps), [batch, 2 + EMBED]);
  const ys = tf.tensor2d(noise, [batch, 2]);
  try {
    const loss = await model.trainOnBatch(xs, ys);
    return Array.isArray(loss) ? loss[0] : loss;
  } finally {
    tf.dispose([xs, ys]);
  }
}

/** The predicted noise for points `x`, all at step t. */
export function predictNoise(model: tf.LayersModel, x: Float32Array, t: number, steps: number): Float32Array {
  return tf.tidy(() => (model.predict(tf.tensor2d(inputs(x, t, steps), [x.length / 2, 2 + EMBED])) as tf.Tensor).dataSync() as Float32Array);
}

/**
 * Generate `count` points: start from N(0, I) and run every reverse step. Returns the
 * points after each step: frames[0] is pure noise (t = T), frames[T] the samples.
 */
export function sample(model: tf.LayersModel, schedule: Schedule, count: number, random: () => number): Float32Array[] {
  const steps = schedule.betas.length;
  let x: Float32Array = Float32Array.from({ length: count * 2 }, () => gaussianRandom(random));
  const frames = [x];
  for (let t = steps - 1; t >= 0; t--) {
    x = reverseStep(x, predictNoise(model, x, t, steps), t, schedule, random);
    frames.push(x);
  }
  return frames;
}
