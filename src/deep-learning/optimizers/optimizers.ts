/**
 * Optimizers on 2-D loss surfaces: how SGD, momentum, Nesterov, RMSprop and Adam move.
 *
 * Every optimizer turns the gradient g of the loss into a step; they differ in what
 * they remember. The update rules are written exactly as PyTorch implements them
 * (torch.optim), so the same numbers come out of optimizers.py:
 *   SGD       p ← p − η·g
 *   Momentum  b ← μ·b + g,                         p ← p − η·b
 *   Nesterov  b ← μ·b + g,                         p ← p − η·(g + μ·b)
 *   RMSprop   v ← α·v + (1 − α)·g²,                p ← p − η·g / (√v + ε)
 *   Adam      m ← β₁·m + (1 − β₁)·g,  v ← β₂·v + (1 − β₂)·g²
 *             p ← p − η·m̂ / (√v̂ + ε),  m̂ = m / (1 − β₁ᵗ), v̂ = v / (1 − β₂ᵗ)
 * Momentum accumulates a velocity, which speeds it up along a consistent direction and
 * averages out zig-zags. RMSprop and Adam divide by a running RMS of the gradient, per
 * coordinate, so steep and flat directions get similar step sizes.
 */

export interface Surface {
  label: string;
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => [number, number];
  domain: { xMin: number; xMax: number; yMin: number; yMax: number };
  minimum: [number, number];
  start: [number, number];
  /** Enough steps to see every optimizer arrive. */
  steps: number;
  /** Learning rates that work reasonably on this surface, per optimizer. */
  rates: Record<OptimizerName, number>;
}

export const OPTIMIZERS = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam'] as const;
export type OptimizerName = (typeof OPTIMIZERS)[number];

export const OPTIMIZER_LABELS: Record<OptimizerName, string> = {
  sgd: 'SGD', momentum: 'Momentum', nesterov: 'Nesterov', rmsprop: 'RMSprop', adam: 'Adam',
};

export const SURFACES = {
  ravine: {
    label: 'Ravine (ill-conditioned)',
    // Curvature 0.2 along x and 4 along y: a long, narrow valley.
    f: (x: number, y: number) => 0.1 * x * x + 2 * y * y,
    grad: (x: number, y: number): [number, number] => [0.2 * x, 4 * y],
    domain: { xMin: -5, xMax: 5, yMin: -2.5, yMax: 2.5 },
    minimum: [0, 0],
    start: [-4.5, 1.6],
    steps: 200,
    rates: { sgd: 0.4, momentum: 0.05, nesterov: 0.05, rmsprop: 0.05, adam: 0.2 },
  },
  rosenbrock: {
    label: 'Rosenbrock (curved valley)',
    f: (x: number, y: number) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    grad: (x: number, y: number): [number, number] => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    domain: { xMin: -2, xMax: 2, yMin: -1, yMax: 3 },
    minimum: [1, 1],
    start: [-1.5, 2.5],
    steps: 1500,
    rates: { sgd: 0.0015, momentum: 0.0003, nesterov: 0.0003, rmsprop: 0.005, adam: 0.05 },
  },
  saddle: {
    label: 'Saddle (two minima)',
    // Minima at (±1, 0), a saddle point at the origin.
    f: (x: number, y: number) => x ** 4 / 4 - x * x / 2 + y * y / 2 + 0.25,
    grad: (x: number, y: number): [number, number] => [x ** 3 - x, y],
    domain: { xMin: -2, xMax: 2, yMin: -2, yMax: 2 },
    minimum: [1, 0],
    start: [0.001, 1.8],
    steps: 300,
    rates: { sgd: 0.1, momentum: 0.02, nesterov: 0.02, rmsprop: 0.02, adam: 0.05 },
  },
} satisfies Record<string, Surface>;
export type SurfaceName = keyof typeof SURFACES;

export interface Hyper {
  momentum: number;
  /** RMSprop's α (PyTorch's default 0.99). */
  alpha: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export const DEFAULT_HYPER: Hyper = { momentum: 0.9, alpha: 0.99, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

interface State { p: number[]; b: number[]; m: number[]; v: number[]; t: number }

/** One update of `name` from the gradient g. */
export function step(name: OptimizerName, s: State, g: number[], lr: number, h: Hyper): State {
  const t = s.t + 1;
  const p = [...s.p], b = [...s.b], m = [...s.m], v = [...s.v];
  for (let i = 0; i < p.length; i++) {
    switch (name) {
      case 'sgd':
        p[i] -= lr * g[i];
        break;
      case 'momentum':
      case 'nesterov':
        b[i] = t === 1 ? g[i] : h.momentum * b[i] + g[i];
        p[i] -= lr * (name === 'nesterov' ? g[i] + h.momentum * b[i] : b[i]);
        break;
      case 'rmsprop':
        v[i] = h.alpha * v[i] + (1 - h.alpha) * g[i] * g[i];
        p[i] -= (lr * g[i]) / (Math.sqrt(v[i]) + h.eps);
        break;
      case 'adam': {
        m[i] = h.beta1 * m[i] + (1 - h.beta1) * g[i];
        v[i] = h.beta2 * v[i] + (1 - h.beta2) * g[i] * g[i];
        const mHat = m[i] / (1 - h.beta1 ** t), vHat = v[i] / (1 - h.beta2 ** t);
        p[i] -= (lr * mHat) / (Math.sqrt(vHat) + h.eps);
        break;
      }
    }
  }
  return { p, b, m, v, t };
}

export interface Run {
  path: [number, number][];
  losses: number[];
  /** True if the iterate blew up (non-finite or far outside the plot). */
  diverged: boolean;
}

/**
 * Run `steps` updates from `start`. With `noise` > 0 each gradient gets Gaussian noise of
 * that standard deviation, like the gradient of a random mini-batch.
 */
export function runOptimizer(
  surface: Surface, name: OptimizerName, lr: number, steps: number, start: [number, number],
  { hyper = DEFAULT_HYPER, noise = 0, random }: { hyper?: Hyper; noise?: number; random?: () => number } = {},
): Run {
  let s: State = { p: [...start], b: [0, 0], m: [0, 0], v: [0, 0], t: 0 };
  const path: [number, number][] = [[start[0], start[1]]];
  const losses = [surface.f(start[0], start[1])];
  const { xMin, xMax, yMin, yMax } = surface.domain;
  const span = Math.max(xMax - xMin, yMax - yMin);
  for (let k = 0; k < steps; k++) {
    const g = surface.grad(s.p[0], s.p[1]);
    if (noise > 0 && random) for (let i = 0; i < 2; i++) g[i] += noise * Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    s = step(name, s, g, lr, hyper);
    const [x, y] = s.p;
    const far = Math.abs(x - (xMin + xMax) / 2) > 10 * span || Math.abs(y - (yMin + yMax) / 2) > 10 * span;
    if (!Number.isFinite(x) || !Number.isFinite(y) || far) return { path, losses, diverged: true };
    path.push([x, y]);
    losses.push(surface.f(x, y));
  }
  return { path, losses, diverged: false };
}
