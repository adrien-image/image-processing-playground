/**
 * An autoencoder squeezes each 784-pixel digit through a 2-number bottleneck and
 * reconstructs it. Because the code has only two numbers, it can be drawn: every digit
 * becomes a point in a plane, and every point of the plane decodes to an image.
 *
 *   encoder   x (784) → 256 (ReLU) → μ (2), log σ² (2)
 *   decoder   z (2)   → 256 (ReLU) → x̂ (784, sigmoid)
 *
 * Plain autoencoder: z = μ, loss = reconstruction error only.
 * Variational autoencoder (VAE): z = μ + σ·ε with ε ~ N(0, 1) (the "reparameterisation
 * trick", so gradients flow through the sampling), and the loss adds β·KL(q(z|x) ‖ N(0, I)),
 * which pulls the codes towards a unit Gaussian. The plane then has no holes: points
 * between digits decode to plausible in-between shapes.
 */
import * as tf from '@tensorflow/tfjs';

const PIXELS = 784;
const HIDDEN = 256;
export const LATENT = 2;

/** KL(N(μ, σ²) ‖ N(0, 1)) summed over the latent dimensions, per row: −½ Σ (1 + log σ² − μ² − σ²). */
export function klDivergence(mu: tf.Tensor2D, logVar: tf.Tensor2D): tf.Tensor1D {
  return tf.mul(-0.5, tf.sum(tf.sub(tf.sub(tf.add(1, logVar), tf.square(mu)), tf.exp(logVar)), 1));
}

const dense = (inputs: number, outputs: number, seed: number) => ({
  w: tf.variable(tf.randomNormal([inputs, outputs], 0, Math.sqrt(2 / (inputs + outputs)), 'float32', seed)),
  b: tf.variable(tf.zeros([outputs])),
});

export interface LossParts {
  total: tf.Scalar;
  reconstruction: tf.Scalar;
  kl: tf.Scalar;
}

export class Autoencoder {
  private readonly enc = dense(PIXELS, HIDDEN, 1);
  private readonly mu = dense(HIDDEN, LATENT, 2);
  private readonly logVar = dense(HIDDEN, LATENT, 3);
  private readonly dec = dense(LATENT, HIDDEN, 4);
  private readonly out = dense(HIDDEN, PIXELS, 5);
  private seed = 100;

  /** Every trainable variable. */
  get variables(): tf.Variable[] {
    return [this.enc, this.mu, this.logVar, this.dec, this.out].flatMap(l => [l.w, l.b]);
  }

  encode(x: tf.Tensor2D): { mu: tf.Tensor2D; logVar: tf.Tensor2D } {
    const h = tf.relu(tf.add(tf.matMul(x, this.enc.w), this.enc.b));
    return {
      mu: tf.add(tf.matMul(h, this.mu.w), this.mu.b),
      // Clipped so exp(log σ²) can never overflow early in training.
      logVar: tf.clipByValue(tf.add(tf.matMul(h, this.logVar.w), this.logVar.b), -10, 10),
    };
  }

  /** Logits of the reconstruction (apply a sigmoid for pixel values). */
  decodeLogits(z: tf.Tensor2D): tf.Tensor2D {
    const h = tf.relu(tf.add(tf.matMul(z, this.dec.w), this.dec.b));
    return tf.add(tf.matMul(h, this.out.w), this.out.b);
  }

  /**
   * Mean loss over a batch: reconstruction (binary cross-entropy summed over the 784
   * pixels) plus, for a VAE, β times the KL term.
   */
  loss(x: tf.Tensor2D, variational: boolean, beta: number): LossParts {
    const { mu, logVar } = this.encode(x);
    const z = variational ? tf.add(mu, tf.mul(tf.exp(tf.mul(0.5, logVar)), tf.randomNormal(mu.shape, 0, 1, 'float32', this.seed++))) : mu;
    const logits = this.decodeLogits(z as tf.Tensor2D);
    // BCE with logits, written out: max(l, 0) − l·x + log(1 + e^−|l|)
    const bce = tf.add(tf.sub(tf.relu(logits), tf.mul(logits, x)), tf.log1p(tf.exp(tf.neg(tf.abs(logits)))));
    const reconstruction = tf.mean(tf.sum(bce, 1)) as tf.Scalar;
    const kl = variational ? (tf.mean(klDivergence(mu, logVar)) as tf.Scalar) : tf.scalar(0);
    return { total: tf.add(reconstruction, tf.mul(beta, kl)) as tf.Scalar, reconstruction, kl };
  }

  /** The 2-D code (μ) of each of `count` images. */
  codes(images: Float32Array, count: number): Float32Array {
    return tf.tidy(() => this.encode(tf.tensor2d(images.subarray(0, count * PIXELS), [count, PIXELS])).mu.dataSync() as Float32Array);
  }

  /** Images (784 values in [0, 1] each) decoded from `points` (pairs of latent coordinates). */
  decode(points: Float32Array): Float32Array {
    const n = points.length / LATENT;
    return tf.tidy(() => tf.sigmoid(this.decodeLogits(tf.tensor2d(points, [n, LATENT]))).dataSync() as Float32Array);
  }

  /** Reconstructions of `count` images through the bottleneck (using μ). */
  reconstruct(images: Float32Array, count: number): Float32Array {
    return this.decode(this.codes(images, count));
  }

  dispose(): void {
    this.variables.forEach(v => v.dispose());
  }
}
