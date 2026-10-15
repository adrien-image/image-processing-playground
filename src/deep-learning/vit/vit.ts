/**
 * A tiny Vision Transformer (ViT) for 28×28 digits, written out with plain tensor ops
 * so every step is visible:
 *
 *   1. patchify    cut the image into 16 patches of 7×7 = 49 pixels      [16 × 49]
 *   2. embed       one linear layer per patch → a D-number "token"       [16 × D]
 *   3. + [CLS]     prepend a learned summary token; add learned position embeddings [17 × D]
 *   4. blocks      L × (LayerNorm → multi-head self-attention → +, LayerNorm → MLP → +)
 *   5. head        the [CLS] token after the last block → 10 class scores
 *
 * Self-attention lets every token look at every other one: token i builds a query,
 * every token j a key, and i takes a weighted average of the tokens' values with weights
 * softmax(q_i · k_j / √d). Unlike a convolution there is no built-in notion of "nearby":
 * the position embeddings are the only clue to where a patch came from.
 */
import * as tf from '@tensorflow/tfjs';

export const PATCH = 7;
export const GRID = 4;
export const PATCHES = GRID * GRID;
export const TOKENS = PATCHES + 1;
const PATCH_DIM = PATCH * PATCH;

/** Rearrange `count` 28×28 images into [count, 16, 49]: patches row by row, pixels row by row inside each. */
export function patchify(images: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count * PATCHES * PATCH_DIM);
  for (let n = 0; n < count; n++) {
    for (let p = 0; p < PATCHES; p++) {
      const py = Math.floor(p / GRID) * PATCH, px = (p % GRID) * PATCH;
      for (let y = 0; y < PATCH; y++) {
        for (let x = 0; x < PATCH; x++) {
          out[(n * PATCHES + p) * PATCH_DIM + y * PATCH + x] = images[n * 784 + (py + y) * 28 + px + x];
        }
      }
    }
  }
  return out;
}

export interface VitConfig {
  dim: number;
  heads: number;
  layers: number;
  mlp: number;
}

export const DEFAULT_CONFIG: VitConfig = { dim: 32, heads: 2, layers: 2, mlp: 64 };

type Linear = { w: tf.Variable; b: tf.Variable };
interface Block { ln1g: tf.Variable; ln1b: tf.Variable; q: Linear; k: Linear; v: Linear; o: Linear; ln2g: tf.Variable; ln2b: tf.Variable; fc1: Linear; fc2: Linear }

let seed = 1;
const linear = (i: number, o: number): Linear => ({
  w: tf.variable(tf.randomNormal([i, o], 0, Math.sqrt(2 / (i + o)), 'float32', seed++)),
  b: tf.variable(tf.zeros([o])),
});

/** x [B, T, in] · W [in, out] + b → [B, T, out] */
function apply(x: tf.Tensor3D, l: Linear): tf.Tensor3D {
  const [b, t, d] = x.shape;
  return tf.reshape(tf.add(tf.matMul(tf.reshape(x, [b * t, d]), l.w), l.b), [b, t, l.w.shape[1] as number]);
}

function layerNorm(x: tf.Tensor3D, gamma: tf.Variable, beta: tf.Variable): tf.Tensor3D {
  const { mean, variance } = tf.moments(x, -1, true);
  return tf.add(tf.mul(tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, 1e-5))), gamma), beta);
}

export interface Forward {
  logits: tf.Tensor2D;
  /** Per block: attention weights [B, heads, 17, 17]; each row sums to 1. */
  attention: tf.Tensor4D[];
}

export class TinyViT {
  readonly config: VitConfig;
  private readonly embed: Linear;
  private readonly cls: tf.Variable;
  private readonly pos: tf.Variable;
  private readonly blocks: Block[];
  private readonly lnfg: tf.Variable;
  private readonly lnfb: tf.Variable;
  private readonly head: Linear;

  constructor(config: VitConfig = DEFAULT_CONFIG) {
    this.config = config;
    const { dim, mlp, layers } = config;
    this.embed = linear(PATCH_DIM, dim);
    this.cls = tf.variable(tf.randomNormal([1, 1, dim], 0, 0.02, 'float32', seed++));
    this.pos = tf.variable(tf.randomNormal([1, TOKENS, dim], 0, 0.02, 'float32', seed++));
    this.blocks = Array.from({ length: layers }, () => ({
      ln1g: tf.variable(tf.ones([dim])), ln1b: tf.variable(tf.zeros([dim])),
      q: linear(dim, dim), k: linear(dim, dim), v: linear(dim, dim), o: linear(dim, dim),
      ln2g: tf.variable(tf.ones([dim])), ln2b: tf.variable(tf.zeros([dim])),
      fc1: linear(dim, mlp), fc2: linear(mlp, dim),
    }));
    this.lnfg = tf.variable(tf.ones([dim]));
    this.lnfb = tf.variable(tf.zeros([dim]));
    this.head = linear(dim, 10);
  }

  get variables(): tf.Variable[] {
    const lin = (l: Linear) => [l.w, l.b];
    return [
      ...lin(this.embed), this.cls, this.pos,
      ...this.blocks.flatMap(b => [b.ln1g, b.ln1b, ...lin(b.q), ...lin(b.k), ...lin(b.v), ...lin(b.o), b.ln2g, b.ln2b, ...lin(b.fc1), ...lin(b.fc2)]),
      this.lnfg, this.lnfb, ...lin(this.head),
    ];
  }

  private attend(x: tf.Tensor3D, b: Block): { out: tf.Tensor3D; weights: tf.Tensor4D } {
    const [batch, tokens, dim] = x.shape;
    const { heads } = this.config;
    const dh = dim / heads;
    // [B, T, D] → [B·H, T, dh]: each head works on its own slice of the features.
    const split = (t: tf.Tensor3D) => tf.reshape(tf.transpose(tf.reshape(t, [batch, tokens, heads, dh]), [0, 2, 1, 3]), [batch * heads, tokens, dh]) as tf.Tensor3D;
    const q = split(apply(x, b.q)), k = split(apply(x, b.k)), v = split(apply(x, b.v));
    const weights = tf.softmax(tf.div(tf.matMul(q, k, false, true), Math.sqrt(dh))) as tf.Tensor3D; // [B·H, T, T]
    const mixed = tf.matMul(weights, v); // [B·H, T, dh]
    const merged = tf.reshape(tf.transpose(tf.reshape(mixed, [batch, heads, tokens, dh]), [0, 2, 1, 3]), [batch, tokens, dim]) as tf.Tensor3D;
    return { out: apply(merged, b.o), weights: tf.reshape(weights, [batch, heads, tokens, tokens]) };
  }

  /** patches: [B, 16, 49] */
  forward(patches: tf.Tensor3D): Forward {
    const batch = patches.shape[0];
    const tokens = apply(patches, this.embed);
    let x = tf.add(tf.concat([tf.tile(this.cls, [batch, 1, 1]), tokens], 1), this.pos) as tf.Tensor3D;
    const attention: tf.Tensor4D[] = [];
    for (const b of this.blocks) {
      const a = this.attend(layerNorm(x, b.ln1g, b.ln1b), b);
      attention.push(a.weights);
      x = tf.add(x, a.out);
      x = tf.add(x, apply(tf.relu(apply(layerNorm(x, b.ln2g, b.ln2b), b.fc1)), b.fc2));
    }
    const clsOut = tf.squeeze(tf.slice(layerNorm(x, this.lnfg, this.lnfb), [0, 0, 0], [batch, 1, this.config.dim]), [1]) as tf.Tensor2D;
    return { logits: tf.add(tf.matMul(clsOut, this.head.w), this.head.b), attention };
  }

  /** Mean cross-entropy of a batch. */
  loss(patches: tf.Tensor3D, labels: tf.Tensor2D): tf.Scalar {
    return tf.losses.softmaxCrossEntropy(labels, this.forward(patches).logits) as tf.Scalar;
  }

  /** Class probabilities and attention maps for `count` images (28×28, values 0–1). */
  run(images: Float32Array, count: number): { probs: Float32Array; attention: Float32Array[] } {
    return tf.tidy(() => {
      const f = this.forward(tf.tensor3d(patchify(images, count), [count, PATCHES, PATCH_DIM]));
      return { probs: tf.softmax(f.logits).dataSync() as Float32Array, attention: f.attention.map(a => a.dataSync() as Float32Array) };
    });
  }

  /** Accuracy on a labelled set, in batches. */
  accuracy(images: Float32Array, labels: Uint8Array, count: number): number {
    let correct = 0;
    for (let start = 0; start < count; start += 250) {
      const n = Math.min(250, count - start);
      const predicted = tf.tidy(() =>
        this.forward(tf.tensor3d(patchify(images.subarray(start * 784, (start + n) * 784), n), [n, PATCHES, PATCH_DIM])).logits.argMax(1).dataSync());
      for (let i = 0; i < n; i++) if (predicted[i] === labels[start + i]) correct++;
    }
    return correct / count;
  }

  /** Cosine similarity between the position embeddings of every pair of patches: [16, 16]. */
  positionSimilarity(): Float32Array {
    return tf.tidy(() => {
      const p = tf.slice(tf.squeeze(this.pos, [0]), [1, 0], [PATCHES, this.config.dim]);
      const unit = tf.div(p, tf.norm(p, 'euclidean', 1, true));
      return tf.matMul(unit, unit, false, true).dataSync() as Float32Array;
    });
  }

  dispose(): void {
    this.variables.forEach(v => v.dispose());
  }
}

/**
 * Attention rollout (Abnar & Zuidema, 2020): how much each patch contributes to the
 * [CLS] token through all the blocks. Per block, average the heads, add the residual
 * connection (½A + ½I), renormalise the rows, and multiply the blocks together.
 * `perBlock` holds one [heads, 17, 17] array per block, for one image.
 */
export function attentionRollout(perBlock: Float32Array[], heads: number): Float32Array {
  const T = TOKENS;
  let rollout = Float64Array.from({ length: T * T }, (_, i) => (Math.floor(i / T) === i % T ? 1 : 0));
  for (const attn of perBlock) {
    const a = new Float64Array(T * T);
    for (let h = 0; h < heads; h++) for (let i = 0; i < T * T; i++) a[i] += attn[h * T * T + i] / heads;
    for (let r = 0; r < T; r++) {
      let sum = 0;
      for (let c = 0; c < T; c++) { a[r * T + c] = 0.5 * a[r * T + c] + (r === c ? 0.5 : 0); sum += a[r * T + c]; }
      for (let c = 0; c < T; c++) a[r * T + c] /= sum;
    }
    const next = new Float64Array(T * T);
    for (let r = 0; r < T; r++) for (let k = 0; k < T; k++) {
      const v = a[r * T + k];
      if (v) for (let c = 0; c < T; c++) next[r * T + c] += v * rollout[k * T + c];
    }
    rollout = next;
  }
  return Float32Array.from(rollout.subarray(1, T)); // [CLS] row, patch columns
}
