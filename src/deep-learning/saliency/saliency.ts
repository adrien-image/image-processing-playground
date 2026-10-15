/**
 * Saliency maps: which pixels does a prediction depend on?
 *
 *   vanillaGradient — |∂ log p(class) / ∂ pixel|: how sensitive the output is to each pixel
 *   smoothGrad      — the same, averaged over noisy copies of the input (less speckled)
 *   gradCam         — weights the last conv layer's feature maps by how much the class
 *                     score depends on them, then upsamples the coarse map to the input
 * (Occlusion, the third family, is in classification/insight.ts.)
 *
 * Each function returns a plain 28×28 Float32Array and frees every tensor it creates.
 */
import * as tf from '@tensorflow/tfjs';
import { gaussianRandom } from '../../lib/random';

const SIZE = 28;
const PIXELS = SIZE * SIZE;

/** log p(class) for a batch, summed (so one gradient pass covers every image). */
const logProb = (model: tf.LayersModel, classIndex: number) => (images: tf.Tensor) => {
  const probs = model.apply(images, { training: false }) as tf.Tensor2D;
  return tf.sum(tf.log(tf.add(probs.slice([0, classIndex], [-1, 1]), 1e-7)));
};

/** |∂ log p(class) / ∂x| for one image. */
export function vanillaGradient(model: tf.LayersModel, image: Float32Array, classIndex: number): Float32Array {
  return tf.tidy(() => tf.abs(tf.grad(logProb(model, classIndex))(tf.tensor4d(image, [1, SIZE, SIZE, 1]))).dataSync() as Float32Array);
}

/**
 * SmoothGrad: the mean absolute gradient over `samples` copies of the image with
 * Gaussian noise (standard deviation `sigma`) added, computed as one batch.
 */
export function smoothGrad(
  model: tf.LayersModel, image: Float32Array, classIndex: number, random: () => number, samples = 24, sigma = 0.15,
): Float32Array {
  const batch = new Float32Array(samples * PIXELS);
  for (let s = 0; s < samples; s++) for (let i = 0; i < PIXELS; i++) batch[s * PIXELS + i] = image[i] + sigma * gaussianRandom(random);
  return tf.tidy(() => {
    const g = tf.abs(tf.grad(logProb(model, classIndex))(tf.tensor4d(batch, [samples, SIZE, SIZE, 1])));
    return tf.mean(g, 0).dataSync() as Float32Array;
  });
}

interface CamModels {
  features: tf.LayersModel;
  head: tf.LayersModel;
}
const camModels = new WeakMap<tf.LayersModel, CamModels>();

/**
 * Split LeNet at its last convolution: `features` maps an image to conv2's 8×8×16
 * activations; `head` maps those to the class probabilities, reusing the same layers
 * (and weights). Cached per model and never disposed, since they share its weights.
 */
function splitAtConv2(model: tf.LayersModel): CamModels {
  let m = camModels.get(model);
  if (!m) {
    const conv2 = model.getLayer('conv2');
    const features = tf.model({ inputs: model.inputs, outputs: conv2.output as tf.SymbolicTensor });
    const input = tf.input({ shape: (conv2.outputShape as number[]).slice(1) });
    let x: tf.SymbolicTensor = input;
    for (const name of ['pool2', 'flatten', 'fc1', 'fc2', 'output']) x = model.getLayer(name).apply(x) as tf.SymbolicTensor;
    m = { features, head: tf.model({ inputs: input, outputs: x }) };
    camModels.set(model, m);
  }
  return m;
}

/**
 * Grad-CAM on conv2. For each of its 16 feature maps A_k, the weight α_k is the mean
 * gradient of log p(class) over the map; the map is ReLU(Σ α_k A_k), 8×8, which is then
 * upsampled to the 28×28 input with `upsampleCam`.
 */
export function gradCam(model: tf.LayersModel, image: Float32Array, classIndex: number): { coarse: Float32Array; map: Float32Array } {
  const { features, head } = splitAtConv2(model);
  const coarse = tf.tidy(() => {
    const a = features.predict(tf.tensor4d(image, [1, SIZE, SIZE, 1])) as tf.Tensor4D;
    const grads = tf.grad(logProb(head, classIndex))(a);
    const alpha = tf.mean(grads, [1, 2], true); // [1, 1, 1, 16]
    return tf.relu(tf.sum(tf.mul(a, alpha), 3)).dataSync() as Float32Array; // [1, 8, 8]
  });
  return { coarse, map: upsampleCam(coarse, 8, CONV2_START, CONV2_JUMP, SIZE) };
}

/**
 * Where conv2 sits on the input: unit 0 sees input pixels 0–13, so its centre is at 7.0 in
 * continuous coordinates (pixel x spans [x, x + 1]); neighbouring units are 2 pixels apart.
 * (receptiveField.ts gives the same numbers, counted in pixel indices: start 6.5, jump 2.)
 */
export const CONV2_START = 7;
export const CONV2_JUMP = 2;

/**
 * Bilinearly resample an n×n map whose unit (i, j) is centred at input coordinate
 * start + jump·(i, j) onto an out×out grid (pixel x is centred at x + 0.5).
 * Positions outside the first/last unit take the edge value.
 */
export function upsampleCam(cam: ArrayLike<number>, n: number, start: number, jump: number, out: number): Float32Array {
  const result = new Float32Array(out * out);
  const coord = (p: number) => Math.min(n - 1, Math.max(0, (p + 0.5 - start) / jump));
  for (let y = 0; y < out; y++) {
    const v = coord(y), y0 = Math.floor(v), y1 = Math.min(n - 1, y0 + 1), fy = v - y0;
    for (let x = 0; x < out; x++) {
      const u = coord(x), x0 = Math.floor(u), x1 = Math.min(n - 1, x0 + 1), fx = u - x0;
      const top = cam[y0 * n + x0] * (1 - fx) + cam[y0 * n + x1] * fx;
      const bottom = cam[y1 * n + x0] * (1 - fx) + cam[y1 * n + x1] * fx;
      result[y * out + x] = top * (1 - fy) + bottom * fy;
    }
  }
  return result;
}
