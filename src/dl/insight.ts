/**
 * Looking inside a trained LeNet (see lenet.ts):
 *
 *  1. conv1Filters   — the 5×5 kernels the first layer has learned,
 *  2. activations    — the feature maps every conv / pool layer produces for one input,
 *  3. occlusionMap   — which pixels the prediction relies on, found by hiding
 *                      small patches of the input and watching the probability drop.
 *
 * Each function copies what it needs out of TensorFlow.js into plain Float32Arrays
 * and disposes every tensor it creates, so the caller has no GPU memory to manage.
 */
import * as tf from '@tensorflow/tfjs';
import { hotColor } from '../hough/hough';

const SIZE = 28;
const PIXELS = SIZE * SIZE;

/** The layers we visualise, in network order (named in `buildLeNet`). */
export const FEATURE_LAYERS = ['conv1', 'pool1', 'conv2', 'pool2'] as const;
export type FeatureLayer = (typeof FEATURE_LAYERS)[number];

// ---------------------------------------------------------------- filters

/**
 * The first convolution layer's kernels: one 5×5 Float32Array per filter (row-major,
 * index y*5+x). They play the role of the hand-made kernels of the Convolution
 * section, except that their 25 weights were learned from the training digits.
 */
export function conv1Filters(model: tf.LayersModel): Float32Array[] {
  // getWeights() hands back the layer's live variables (not copies): read, never dispose.
  // TensorFlow.js stores a conv kernel as [height, width, inChannels, outChannels] = [5, 5, 1, 6].
  const [kernel] = model.getLayer('conv1').getWeights();
  const [kh, kw, inChannels, filters] = kernel.shape;
  const weights = kernel.dataSync();
  return Array.from({ length: filters }, (_, f) => {
    const filter = new Float32Array(kh * kw);
    for (let y = 0; y < kh; y++) {
      for (let x = 0; x < kw; x++) {
        // input channel 0: the digits are grayscale
        filter[y * kw + x] = weights[(y * kw + x) * inChannels * filters + f];
      }
    }
    return filter;
  });
}

// ---------------------------------------------------------------- activations

export interface FeatureMaps {
  name: FeatureLayer;
  height: number;
  width: number;
  channels: number;
  /** Channel-major: channel c at (x, y) is data[(c * height + y) * width + x]. */
  data: Float32Array;
}

const probes = new WeakMap<tf.LayersModel, tf.LayersModel>();

/**
 * A second model over the *same* layers (so the same weights) whose outputs are the
 * feature layers instead of the class probabilities. Cached per model, and never
 * disposed: disposing it would dispose the layers it shares with `model`.
 */
function featureModel(model: tf.LayersModel): tf.LayersModel {
  let probe = probes.get(model);
  if (!probe) {
    const outputs = FEATURE_LAYERS.map(name => model.getLayer(name).output as tf.SymbolicTensor);
    probe = tf.model({ inputs: model.inputs, outputs });
    probes.set(model, probe);
  }
  return probe;
}

/**
 * What each feature layer outputs for one 28×28 input (784 values in [0, 1]):
 * conv1 24×24×6, pool1 12×12×6, conv2 8×8×16, pool2 4×4×16.
 * Conv outputs are taken after their ReLU, so every value is ≥ 0.
 */
export function activations(model: tf.LayersModel, input: Float32Array): FeatureMaps[] {
  const probe = featureModel(model);
  // tidy() disposes the input tensor, the layer outputs and every intermediate.
  const outputs = tf.tidy(() => {
    const result = probe.predict(tf.tensor4d(input, [1, SIZE, SIZE, 1])) as tf.Tensor[];
    return result.map(t => t.dataSync() as Float32Array);
  });

  return FEATURE_LAYERS.map((name, i) => {
    const [, height, width, channels] = probe.outputs[i].shape as number[];
    // TensorFlow.js is channels-last ([1, h, w, c]); regroup it one channel at a time.
    const nhwc = outputs[i];
    const data = new Float32Array(height * width * channels);
    for (let c = 0; c < channels; c++) {
      for (let p = 0; p < height * width; p++) data[c * height * width + p] = nhwc[p * channels + c];
    }
    return { name, height, width, channels, data };
  });
}

// ---------------------------------------------------------------- occlusion

/**
 * Top-left corners of `patch`-wide windows moved by `stride` across `size` pixels,
 * plus a final window flush with the far edge, so every pixel is covered.
 */
export function patchPositions(size: number, patch: number, stride: number): number[] {
  if (patch < 1 || patch > size || stride < 1) throw new RangeError('Need 1 ≤ patch ≤ size and stride ≥ 1');
  const positions: number[] = [];
  for (let p = 0; p + patch <= size; p += stride) positions.push(p);
  if (positions[positions.length - 1] + patch < size) positions.push(size - patch);
  return positions;
}

/**
 * Occlusion sensitivity: hide one patch × patch square of the input at a time (set it
 * to 0, the background) and measure how much the probability of `classIndex` drops.
 * All occluded copies, plus the original, go through the network in ONE batch.
 *
 * Returns a 28×28 map (row-major): for each pixel, the mean drop over all patches
 * that covered it.
 *   positive → hiding these pixels lowers the probability: the prediction relies on them
 *   ≈ 0      → hiding them doesn't matter. A patch that covers only background changes
 *              nothing (its drop is exactly 0); background next to a stroke still gets
 *              colour from the patches that hid the stroke as well.
 *   negative → hiding them *raises* the probability: they argue against the class
 */
export function occlusionMap(
  model: tf.LayersModel,
  input: Float32Array,
  classIndex: number,
  { patch = 6, stride = 2 }: { patch?: number; stride?: number } = {},
): Float32Array {
  const positions = patchPositions(SIZE, patch, stride);
  const patches = positions.flatMap(y => positions.map(x => ({ x, y })));

  // Image 0 is the untouched input; image k (k ≥ 1) has patch k − 1 blanked out.
  const batch = new Float32Array((patches.length + 1) * PIXELS);
  batch.set(input);
  patches.forEach(({ x, y }, k) => {
    const image = batch.subarray((k + 1) * PIXELS, (k + 2) * PIXELS);
    image.set(input);
    for (let row = y; row < y + patch; row++) image.fill(0, row * SIZE + x, row * SIZE + x + patch);
  });

  const images = patches.length + 1;
  const probability = tf.tidy(() => {
    // batchSize: by default predict() would cut the batch into chunks of 32 images.
    const probs = model.predict(tf.tensor4d(batch, [images, SIZE, SIZE, 1]), { batchSize: images }) as tf.Tensor2D;
    return probs.slice([0, classIndex], [images, 1]).dataSync() as Float32Array;
  });

  // Spread each patch's drop over the pixels it hid, then average per pixel.
  const sum = new Float32Array(PIXELS);
  const count = new Uint16Array(PIXELS);
  patches.forEach(({ x, y }, k) => {
    const drop = probability[0] - probability[k + 1];
    for (let row = y; row < y + patch; row++) {
      for (let col = x; col < x + patch; col++) {
        sum[row * SIZE + col] += drop;
        count[row * SIZE + col]++;
      }
    }
  });
  for (let i = 0; i < PIXELS; i++) sum[i] /= count[i];
  return sum;
}

// ---------------------------------------------------------------- colours

const NEGATIVE: readonly [number, number, number] = [70, 140, 255];
const POSITIVE: readonly [number, number, number] = [255, 70, 50];

/**
 * Diverging colour map for signed values t in [−1, 1] (clamped):
 * −1 → blue, 0 → black, +1 → red, fading linearly in between.
 */
export function divergingColor(t: number): [number, number, number] {
  const c = Number.isFinite(t) ? Math.max(-1, Math.min(1, t)) : 0;
  const [r, g, b] = c < 0 ? NEGATIVE : POSITIVE;
  const a = Math.abs(c);
  return [r * a, g * a, b * a];
}

export type OcclusionView = 'heat' | 'signed';

/** How much of the white "veil" that shows the digit is laid over the map at full ink. */
const INK_VEIL = 0.4;

/**
 * Colour of one occlusion-map pixel, for t = value / scale in [−1, 1] and the input's
 * ink (0–1) at that pixel:
 *   'heat'   → drops only: black → red → yellow (hotColor); negative values stay black
 *   'signed' → divergingColor: red = hiding lowers P, blue = hiding raises it
 * The digit is drawn as a light veil on top, so on the background the colour is
 * exactly the legend's colour for t.
 */
export function occlusionColor(t: number, ink: number, view: OcclusionView): [number, number, number] {
  const v = Number.isFinite(t) ? t : 0;
  const [r, g, b] = view === 'heat' ? hotColor(0.85 * Math.max(0, v)) : divergingColor(v);
  const veil = INK_VEIL * Math.min(1, Math.max(0, ink));
  return [r + (255 - r) * veil, g + (255 - g) * veil, b + (255 - b) * veil];
}
