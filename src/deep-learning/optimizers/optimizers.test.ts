import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { DEFAULT_HYPER, OPTIMIZERS, runOptimizer, step, SURFACES } from './optimizers';

const last = <T,>(a: T[]) => a[a.length - 1];

describe('step', () => {
  it("Adam's first step moves every coordinate by about η, whatever the gradient's scale", () => {
    const s = step('adam', { p: [0, 0], b: [0, 0], m: [0, 0], v: [0, 0], t: 0 }, [1000, 0.001], 0.1, DEFAULT_HYPER);
    expect(s.p[0]).toBeCloseTo(-0.1, 6);
    expect(s.p[1]).toBeCloseTo(-0.1, 4);
  });

  it('momentum and Nesterov start like SGD, then accumulate', () => {
    const zero = { p: [0], b: [0], m: [0], v: [0], t: 0 };
    const sgd = step('sgd', zero, [1], 0.1, DEFAULT_HYPER);
    const mom = step('momentum', zero, [1], 0.1, DEFAULT_HYPER);
    expect(mom.p[0]).toBeCloseTo(sgd.p[0], 12);
    expect(step('momentum', mom, [1], 0.1, DEFAULT_HYPER).p[0]).toBeCloseTo(-0.1 - 0.19, 12); // b = 0.9·1 + 1
    expect(step('nesterov', zero, [1], 0.1, DEFAULT_HYPER).p[0]).toBeCloseTo(-0.19, 12);       // g + μ·b
  });
});

describe('runOptimizer', () => {
  it('every optimizer converges on every surface with its default learning rate', () => {
    for (const surface of Object.values(SURFACES)) {
      for (const name of OPTIMIZERS) {
        const run = runOptimizer(surface, name, surface.rates[name], 3000, surface.start);
        expect(run.diverged).toBe(false);
        expect(last(run.losses)).toBeLessThan(surface.f(...surface.start) * 0.05);
      }
    }
  });

  it('momentum crosses the ravine far faster than SGD at the same learning rate', () => {
    const { ravine } = SURFACES;
    const sgd = runOptimizer(ravine, 'sgd', 0.05, 100, ravine.start);
    const mom = runOptimizer(ravine, 'momentum', 0.05, 100, ravine.start);
    expect(last(mom.losses)).toBeLessThan(last(sgd.losses) / 10);
  });

  it('SGD diverges beyond η = 2/curvature, while Adam does not', () => {
    const { ravine } = SURFACES; // curvature 4 along y: the limit is 0.5
    expect(runOptimizer(ravine, 'sgd', 0.45, 300, ravine.start).diverged).toBe(false);
    expect(runOptimizer(ravine, 'sgd', 0.55, 300, ravine.start).diverged).toBe(true);
    expect(runOptimizer(ravine, 'adam', 0.55, 300, ravine.start).diverged).toBe(false);
  });

  it('is reproducible with noise, and noise keeps it from settling exactly', () => {
    const { ravine } = SURFACES;
    const a = runOptimizer(ravine, 'sgd', 0.1, 200, ravine.start, { noise: 1, random: seededRandom(2) });
    const b = runOptimizer(ravine, 'sgd', 0.1, 200, ravine.start, { noise: 1, random: seededRandom(2) });
    expect(a.path).toEqual(b.path);
    expect(last(a.losses)).toBeGreaterThan(last(runOptimizer(ravine, 'sgd', 0.1, 200, ravine.start).losses));
  });
});
