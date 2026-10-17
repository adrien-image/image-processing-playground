/**
 * Generative adversarial networks (Goodfellow et al. 2014), on 2-D points.
 *
 * A generator G turns random noise z into points; a discriminator D outputs the
 * probability that a point is real. They are trained against each other:
 *   D minimises  −log D(x) − log(1 − D(G(z)))      (tell real from generated)
 *   G minimises  −log D(G(z))                       ("non-saturating": fool D)
 * The original minimax objective has G minimise log(1 − D(G(z))) instead. Early on D
 * easily rejects G's points, D(G(z)) ≈ 0, and there that loss is flat: its gradient
 * σ(logit) vanishes, so G barely learns. −log D(G(z)) has gradient 1 − σ ≈ 1 there.
 *
 * Training is a balance, not a descent: G chases wherever D currently says "real", D
 * moves the boundary, and G often settles on a few of the data's modes and ignores the
 * rest (mode collapse), which the ring of eight Gaussians makes easy to see.
 */
import * as tf from '@tensorflow/tfjs';
import { gaussianRandom } from '../../lib/random';

export const NOISE = 2;

export type GanLoss = 'non-saturating' | 'minimax';

/** Eight Gaussians on a circle of radius 1.2: the classic mode-collapse test. */
export const MIXTURE_CENTRES: [number, number][] = Array.from({ length: 8 }, (_, k) => [1.2 * Math.cos((k * Math.PI) / 4), 1.2 * Math.sin((k * Math.PI) / 4)]);

export function sampleMixture(count: number, random: () => number, spread = 0.06): Float32Array {
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const [cx, cy] = MIXTURE_CENTRES[Math.floor(random() * 8)];
    out[2 * i] = cx + spread * gaussianRandom(random);
    out[2 * i + 1] = cy + spread * gaussianRandom(random);
  }
  return out;
}

/** How many of the modes have at least `min` of the samples within `radius` of their centre. */
export function modesCovered(samples: Float32Array, centres: [number, number][], radius = 0.25, min = 5): number {
  return centres.filter(([cx, cy]) => {
    let n = 0;
    for (let i = 0; i < samples.length / 2; i++) if (Math.hypot(samples[2 * i] - cx, samples[2 * i + 1] - cy) < radius) n++;
    return n >= min;
  }).length;
}

function mlp(input: number, output: number, seed: number, activation: 'relu' | 'leakyRelu'): tf.Sequential {
  const init = (s: number) => tf.initializers.glorotUniform({ seed: seed * 10 + s });
  const layers: tf.layers.Layer[] = [];
  for (let i = 0; i < 3; i++) {
    layers.push(tf.layers.dense({ inputShape: i === 0 ? [input] : undefined, units: 64, kernelInitializer: init(i) }));
    layers.push(activation === 'relu' ? tf.layers.reLU() : tf.layers.leakyReLU({ alpha: 0.2 }));
  }
  layers.push(tf.layers.dense({ units: output, kernelInitializer: init(9) }));
  return tf.sequential({ layers });
}

const variablesOf = (m: tf.LayersModel) => m.trainableWeights.map(w => w.read() as tf.Variable);

export class Gan {
  readonly generator = mlp(NOISE, 2, 1, 'relu');
  readonly discriminator = mlp(2, 1, 2, 'leakyRelu');
  private readonly optG: tf.Optimizer;
  private readonly optD: tf.Optimizer;

  constructor(learningRate = 1e-3) {
    // β₁ = 0.5, as is usual for GANs: less momentum keeps the two players from overshooting.
    this.optG = tf.train.adam(learningRate, 0.5);
    this.optD = tf.train.adam(learningRate, 0.5);
  }

  private noise(count: number, random: () => number): tf.Tensor2D {
    return tf.tensor2d(Float32Array.from({ length: count * NOISE }, () => gaussianRandom(random)), [count, NOISE]);
  }

  /** One discriminator update (or several), then one generator update. Returns both losses. */
  step(real: Float32Array, random: () => number, { loss = 'non-saturating', dSteps = 1 }: { loss?: GanLoss; dSteps?: number } = {}): { d: number; g: number } {
    const count = real.length / 2;
    let d = NaN;
    for (let k = 0; k < dSteps; k++) {
      const cost = tf.tidy(() => {
        const x = tf.tensor2d(real, [count, 2]);
        const fake = this.generator.predict(this.noise(count, random)) as tf.Tensor2D;
        return this.optD.minimize(() => {
          const realLogits = this.discriminator.apply(x, { training: true }) as tf.Tensor;
          const fakeLogits = this.discriminator.apply(fake, { training: true }) as tf.Tensor;
          return tf.losses.sigmoidCrossEntropy(tf.onesLike(realLogits), realLogits)
            .add(tf.losses.sigmoidCrossEntropy(tf.zerosLike(fakeLogits), fakeLogits)) as tf.Scalar;
        }, true, variablesOf(this.discriminator)) ?? tf.scalar(NaN);
      });
      d = cost.dataSync()[0];
      cost.dispose();
    }
    const gCost = tf.tidy(() => {
      const z = this.noise(count, random);
      return this.optG.minimize(() => {
        const logits = this.discriminator.apply(this.generator.apply(z, { training: true }), { training: true }) as tf.Tensor;
        return (loss === 'non-saturating'
          ? tf.losses.sigmoidCrossEntropy(tf.onesLike(logits), logits)            // −log D(G(z))
          : tf.losses.sigmoidCrossEntropy(tf.zerosLike(logits), logits).neg()      // log(1 − D(G(z)))
        ) as tf.Scalar;
      }, true, variablesOf(this.generator)) ?? tf.scalar(NaN);
    });
    const g = gCost.dataSync()[0];
    gCost.dispose();
    return { d, g };
  }

  sample(count: number, random: () => number): Float32Array {
    return tf.tidy(() => (this.generator.predict(this.noise(count, random)) as tf.Tensor).dataSync() as Float32Array);
  }

  /** D's probability of "real" on a res × res grid over [lo, hi]², row-major, y up. */
  decisionMap(res: number, lo: number, hi: number): Float32Array {
    const pts = new Float32Array(res * res * 2);
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
      pts[2 * (j * res + i)] = lo + ((i + 0.5) / res) * (hi - lo);
      pts[2 * (j * res + i) + 1] = hi - ((j + 0.5) / res) * (hi - lo);
    }
    return tf.tidy(() => (this.discriminator.predict(tf.tensor2d(pts, [res * res, 2])) as tf.Tensor).sigmoid().dataSync() as Float32Array);
  }

  dispose(): void {
    this.optG.dispose();
    this.optD.dispose();
    this.generator.dispose();
    this.discriminator.dispose();
  }
}
