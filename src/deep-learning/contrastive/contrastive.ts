/**
 * Contrastive learning (SimCLR, Chen et al. 2020): learning features without labels.
 *
 * Take a batch of N images and make two random augmentations of each (shift, rotate,
 * zoom, noise): 2N views. An encoder maps every view to a representation h, and a small
 * projection head maps h to z. The loss asks each view to pick out its partner among the
 * 2N − 1 others:
 *     ℓ(i, j) = −log( exp(sim(zᵢ, zⱼ)/τ) / Σ_{k ≠ i} exp(sim(zᵢ, z_k)/τ) )
 * with sim the cosine similarity and τ a temperature (NT-Xent). Nothing here uses the
 * digit labels: the only "label" is which two views came from the same image. To be
 * found again after any augmentation, a view's representation has to capture what the
 * augmentations do not change, which is mostly what digit it is. The labels are used
 * only afterwards, to measure how good the features are with a k-nearest-neighbour vote.
 */
import * as tf from '@tensorflow/tfjs';

export const REPRESENTATION = 64;
export const PROJECTION = 32;

/**
 * Random augmentation of 28×28 images (values 0–1): rotation ±`degrees`, zoom ×[0.85, 1.15],
 * shift ±`shift` pixels (one inverse affine map, bilinear), then Gaussian noise.
 */
export function augment(images: Float32Array, count: number, random: () => number, { degrees = 15, shift = 3, noise = 0.1 } = {}): Float32Array {
  const out = new Float32Array(count * 784);
  for (let n = 0; n < count; n++) {
    const a = ((random() * 2 - 1) * degrees * Math.PI) / 180;
    const s = 0.85 + 0.3 * random();
    const tx = (random() * 2 - 1) * shift, ty = (random() * 2 - 1) * shift;
    const c = Math.cos(a) / s, si = Math.sin(a) / s;
    const src = images.subarray(n * 784, (n + 1) * 784);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        // Output pixel → input pixel: undo the shift, then rotate and zoom about the centre.
        const dx = x - 13.5 - tx, dy = y - 13.5 - ty;
        const sx = c * dx + si * dy + 13.5, sy = -si * dx + c * dy + 13.5;
        let v = 0;
        if (sx >= 0 && sy >= 0 && sx < 27 && sy < 27) {
          const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
          const i = y0 * 28 + x0;
          v = (src[i] * (1 - fx) + src[i + 1] * fx) * (1 - fy) + (src[i + 28] * (1 - fx) + src[i + 29] * fx) * fy;
        }
        const g = Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
        out[n * 784 + y * 28 + x] = Math.min(1, Math.max(0, v + noise * g));
      }
    }
  }
  return out;
}

/** NT-Xent loss of 2N projections: rows i and i + N are the two views of the same image. */
export function ntXent(z1: tf.Tensor2D, z2: tf.Tensor2D, temperature: number): tf.Scalar {
  return tf.tidy(() => {
    const n = z1.shape[0];
    const z = tf.concat([z1, z2]).div(tf.concat([z1, z2]).norm('euclidean', 1, true).add(1e-8)) as tf.Tensor2D;
    // Cosine similarities / τ, with each view's similarity to itself masked out.
    const logits = tf.matMul(z, z, false, true).div(temperature).sub(tf.eye(2 * n).mul(1e9));
    const partner = tf.concat([tf.range(n, 2 * n, 1, 'int32'), tf.range(0, n, 1, 'int32')]);
    return tf.losses.softmaxCrossEntropy(tf.oneHot(partner, 2 * n), logits) as tf.Scalar;
  });
}

export interface SimClr {
  encoder: tf.Sequential;
  projector: tf.Sequential;
  optimizer: tf.Optimizer;
  variables: tf.Variable[];
}

export function buildSimClr(learningRate = 0.002, seed = 1): SimClr {
  const init = (s: number) => tf.initializers.glorotUniform({ seed: seed * 10 + s });
  const encoder = tf.sequential({
    layers: [
      tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 16, kernelSize: 3, activation: 'relu', kernelInitializer: init(1) }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', kernelInitializer: init(2) }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.flatten(),
      tf.layers.dense({ units: REPRESENTATION, activation: 'relu', kernelInitializer: init(3) }),
    ],
  });
  const projector = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [REPRESENTATION], units: REPRESENTATION, activation: 'relu', kernelInitializer: init(4) }),
      tf.layers.dense({ units: PROJECTION, kernelInitializer: init(5) }),
    ],
  });
  const variables = [...encoder.trainableWeights, ...projector.trainableWeights].map(w => w.read() as tf.Variable);
  return { encoder, projector, optimizer: tf.train.adam(learningRate), variables };
}

/** One step on a batch of `count` unlabelled images: two augmentations each. Returns the loss. */
export function trainStep(m: SimClr, images: Float32Array, count: number, random: () => number, temperature: number): number {
  const a = augment(images, count, random), b = augment(images, count, random);
  const cost = tf.tidy(() => {
    const project = (views: Float32Array) =>
      m.projector.apply(m.encoder.apply(tf.tensor4d(views, [count, 28, 28, 1]), { training: true }), { training: true }) as tf.Tensor2D;
    return m.optimizer.minimize(() => ntXent(project(a), project(b), temperature), true, m.variables) ?? tf.scalar(NaN);
  });
  const value = cost.dataSync()[0];
  cost.dispose();
  return value;
}

/** Representations h (count × 64) of un-augmented images. */
export function embed(m: SimClr, images: Float32Array, count: number): Float32Array {
  return tf.tidy(() => (m.encoder.predict(tf.tensor4d(images, [count, 28, 28, 1])) as tf.Tensor).dataSync() as Float32Array);
}

/** Accuracy of a k-nearest-neighbour vote with cosine similarity. */
export function knnAccuracy(train: Float32Array, trainLabels: ArrayLike<number>, test: Float32Array, testLabels: ArrayLike<number>, d: number, k = 5): number {
  const unit = (v: Float32Array, n: number) => {
    const out = new Float32Array(v.length);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += v[i * d + j] ** 2;
      const inv = 1 / (Math.sqrt(s) || 1);
      for (let j = 0; j < d; j++) out[i * d + j] = v[i * d + j] * inv;
    }
    return out;
  };
  const nTrain = train.length / d, nTest = test.length / d;
  const a = unit(train, nTrain), b = unit(test, nTest);
  let right = 0;
  const sims = new Float32Array(nTrain);
  for (let t = 0; t < nTest; t++) {
    for (let i = 0; i < nTrain; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += a[i * d + j] * b[t * d + j];
      sims[i] = s;
    }
    const top = Array.from(sims.keys()).sort((p, q) => sims[q] - sims[p]).slice(0, k);
    const votes = new Map<number, number>();
    for (const i of top) votes.set(trainLabels[i], (votes.get(trainLabels[i]) ?? 0) + 1);
    let best = -1, bestVotes = 0;
    for (const [label, v] of votes) if (v > bestVotes) { best = label; bestVotes = v; }
    if (best === testLabels[t]) right++;
  }
  return right / nTest;
}

/** The two principal components of n×d data, by power iteration on the covariance: n×2 coordinates. */
export function pca2(x: Float32Array, n: number, d: number): Float32Array {
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += x[i * d + j] / n;
  const centred = Float64Array.from(x, (v, k) => v - mean[k % d]);
  const cov = new Float64Array(d * d);
  for (let i = 0; i < n; i++) for (let p = 0; p < d; p++) for (let q = 0; q < d; q++) cov[p * d + q] += (centred[i * d + p] * centred[i * d + q]) / n;
  const components: Float64Array[] = [];
  for (let c = 0; c < 2; c++) {
    let v = Float64Array.from({ length: d }, (_, j) => Math.sin(j + 1 + c));
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(d);
      for (let p = 0; p < d; p++) for (let q = 0; q < d; q++) w[p] += cov[p * d + q] * v[q];
      for (const u of components) { const dot = w.reduce((s, val, j) => s + val * u[j], 0); for (let j = 0; j < d; j++) w[j] -= dot * u[j]; }
      const norm = Math.hypot(...w) || 1;
      v = w.map(val => val / norm);
    }
    components.push(v);
  }
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += centred[i * d + j] * components[c][j];
    out[i * 2 + c] = s;
  }
  return out;
}

export function disposeSimClr(m: SimClr): void {
  m.optimizer.dispose();
  m.encoder.dispose();
  m.projector.dispose();
}
