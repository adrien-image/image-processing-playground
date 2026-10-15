/**
 * Denoising diffusion (DDPM), the idea behind modern image generators.
 *
 * Forward process (fixed, no learning): add a little Gaussian noise at each of T steps,
 * so data turns into pure noise. Thanks to a closed form we can jump straight to step t:
 *
 *     x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,     ε ~ N(0, I),   ᾱ_t = Π_{s ≤ t} (1 − β_s)
 *
 * Reverse process (learned): a network ε_θ(x_t, t) is trained to predict the noise ε that
 * was added. Starting from pure noise, repeatedly remove the predicted noise:
 *
 *     x_{t−1} = 1/√α_t · (x_t − β_t/√(1 − ᾱ_t) · ε_θ(x_t, t)) + √β_t · z,   z ~ N(0, I)
 *
 * Everything here is plain arithmetic on Float32Arrays; the network is in denoiser.ts.
 */
import { gaussianRandom } from '../../lib/random';

export interface Schedule {
  /** β_t for t = 0 … T−1 (the noise added at step t + 1) */
  betas: Float64Array;
  /** α_t = 1 − β_t */
  alphas: Float64Array;
  /** ᾱ_t = Π α_s for s ≤ t */
  alphaBars: Float64Array;
}

function fromBetas(betas: Float64Array): Schedule {
  const alphas = betas.map(b => 1 - b);
  const alphaBars = new Float64Array(betas.length);
  let product = 1;
  alphas.forEach((a, t) => { product *= a; alphaBars[t] = product; });
  return { betas, alphas, alphaBars };
}

/** β rising linearly from β₁ to β_T (the original DDPM schedule). */
export function linearSchedule(steps: number, start = 1e-4, end = 0.02): Schedule {
  return fromBetas(Float64Array.from({ length: steps }, (_, t) => start + ((end - start) * t) / Math.max(1, steps - 1)));
}

/** The cosine schedule (Nichol & Dhariwal, 2021): ᾱ follows a cosine, destroying information more evenly. */
export function cosineSchedule(steps: number, s = 0.008): Schedule {
  const f = (t: number) => Math.cos(((t / steps + s) / (1 + s)) * (Math.PI / 2)) ** 2;
  return fromBetas(Float64Array.from({ length: steps }, (_, t) => Math.min(0.999, 1 - f(t + 1) / f(t))));
}

/** x_t from x_0 in one jump (t is 0-based: t = 0 is one noising step). */
export function qSample(x0: ArrayLike<number>, t: number, schedule: Schedule, noise: ArrayLike<number>): Float32Array {
  const a = Math.sqrt(schedule.alphaBars[t]), b = Math.sqrt(1 - schedule.alphaBars[t]);
  return Float32Array.from(x0, (v, i) => a * v + b * noise[i]);
}

/** One reverse step from x_t to x_{t−1}, given the predicted noise. At t = 0 no fresh noise is added. */
export function reverseStep(xt: ArrayLike<number>, epsilon: ArrayLike<number>, t: number, schedule: Schedule, random: () => number): Float32Array {
  const { betas, alphas, alphaBars } = schedule;
  const k = betas[t] / Math.sqrt(1 - alphaBars[t]);
  const scale = 1 / Math.sqrt(alphas[t]);
  const sigma = t > 0 ? Math.sqrt(betas[t]) : 0;
  return Float32Array.from(xt, (v, i) => scale * (v - k * epsilon[i]) + (sigma ? sigma * gaussianRandom(random) : 0));
}

/** Sinusoidal embedding of the step number (as in transformers), so the network knows how noisy x_t is. */
export function timeEmbedding(t: number, steps: number, dims = 16): Float32Array {
  const out = new Float32Array(dims);
  const half = dims / 2;
  const x = (t / steps) * 1000;
  for (let k = 0; k < half; k++) {
    const freq = Math.exp((-Math.log(10_000) * k) / half);
    out[k] = Math.sin(x * freq);
    out[half + k] = Math.cos(x * freq);
  }
  return out;
}

// ---------------------------------------------------------------- 2-D toy data

export type ToyShape = 'spiral' | 'moons' | 'rings';

/** `count` points (x, y interleaved) of a 2-D distribution, scaled to roughly [−1.5, 1.5]. */
export function sampleToy(shape: ToyShape, count: number, random: () => number): Float32Array {
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    let x: number, y: number;
    if (shape === 'spiral') {
      const t = Math.sqrt(random()) * 3 * Math.PI;
      x = (t * Math.cos(t)) / 6.5;
      y = (t * Math.sin(t)) / 6.5;
    } else if (shape === 'moons') {
      const a = random() * Math.PI;
      const upper = random() < 0.5;
      x = upper ? Math.cos(a) - 0.5 : 0.5 - Math.cos(a);
      y = upper ? Math.sin(a) - 0.25 : 0.25 - Math.sin(a);
    } else {
      const a = random() * 2 * Math.PI;
      const r = random() < 0.5 ? 0.5 : 1.2;
      x = r * Math.cos(a);
      y = r * Math.sin(a);
    }
    out[2 * i] = x + 0.03 * gaussianRandom(random);
    out[2 * i + 1] = y + 0.03 * gaussianRandom(random);
  }
  return out;
}
