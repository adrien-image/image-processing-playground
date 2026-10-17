import { describe, expect, it } from 'vitest';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { affinities, conditionalP, jointP, klGradient, knnAgreement, nearestNeighbours, neighbourhoodKept, rowPerplexity, squaredDistances, Tsne } from './tsne';

const randomPoints = (n: number, d: number, random: () => number) => Float64Array.from({ length: n * d }, () => gaussianRandom(random));

/** k well-separated Gaussian clusters of `per` points in d dimensions, and their labels. */
function clusters(k: number, per: number, d: number, random: () => number): { X: Float64Array; labels: number[] } {
  const centres = Array.from({ length: k }, () => Array.from({ length: d }, () => 8 * gaussianRandom(random)));
  const X = new Float64Array(k * per * d), labels: number[] = [];
  for (let c = 0; c < k; c++) for (let s = 0; s < per; s++) {
    const r = labels.length;
    labels.push(c);
    for (let i = 0; i < d; i++) X[r * d + i] = centres[c][i] + gaussianRandom(random);
  }
  return { X, labels };
}

describe('affinities', () => {
  it('finds σᵢ so that every row has the target perplexity', () => {
    const n = 80, X = randomPoints(n, 5, seededRandom(1));
    const D = squaredDistances(X, n, 5);
    for (const perplexity of [5, 20, 50]) {
      const { P, sigma } = conditionalP(D, n, perplexity);
      for (let i = 0; i < n; i++) {
        expect(rowPerplexity(P, n, i)).toBeCloseTo(perplexity, 2);
        expect(P[i * n + i]).toBe(0);
        expect(sigma[i]).toBeGreaterThan(0);
      }
    }
  });

  it('gives a wider Gaussian to a point in a sparse region', () => {
    // Points 0…29 tightly packed, point 30 far away from them.
    const random = seededRandom(2);
    const X = Float64Array.from({ length: 31 * 2 }, (_, i) => (i < 60 ? 0.1 * gaussianRandom(random) : 5));
    const { sigma } = conditionalP(squaredDistances(X, 31, 2), 31, 10);
    expect(sigma[30]).toBeGreaterThan(5 * sigma[0]);
  });

  it('symmetrises P into a joint distribution that sums to 1', () => {
    const n = 50, P = affinities(randomPoints(n, 4, seededRandom(3)), n, 4, 15);
    let sum = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      expect(P[i * n + j]).toBeCloseTo(P[j * n + i], 15);
      sum += P[i * n + j];
    }
    expect(sum).toBeCloseTo(1, 9);
    // jointP of an asymmetric conditional matrix is still symmetric.
    const C = Float64Array.from([0, 1, 0, 0.5, 0, 0.5, 0.2, 0.8, 0]);
    const J = jointP(C, 3);
    expect(J[1]).toBeCloseTo(J[3], 15);
    expect(J[1]).toBeCloseTo((1 + 0.5) / 6, 9);
  });
});

describe('klGradient', () => {
  it('matches a finite-difference gradient of KL(P ‖ Q)', () => {
    const n = 7, random = seededRandom(4);
    const P = affinities(randomPoints(n, 3, random), n, 3, 3);
    const Y = randomPoints(n, 2, random);
    const { kl, grad } = klGradient(P, Y, n);
    // KL against the direct formula Σ p log(p / q).
    let Z = 0;
    const w = (i: number, j: number) => 1 / (1 + (Y[2 * i] - Y[2 * j]) ** 2 + (Y[2 * i + 1] - Y[2 * j + 1]) ** 2);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) Z += w(i, j);
    let direct = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) direct += P[i * n + j] * Math.log(P[i * n + j] / (w(i, j) / Z));
    expect(kl).toBeCloseTo(direct, 10);
    const h = 1e-6;
    for (let c = 0; c < 2 * n; c++) {
      const plus = Float64Array.from(Y), minus = Float64Array.from(Y);
      plus[c] += h; minus[c] -= h;
      const fd = (klGradient(P, plus, n).kl - klGradient(P, minus, n).kl) / (2 * h);
      expect(grad[c]).toBeCloseTo(fd, 6);
    }
  });
});

describe('Tsne', () => {
  it.each([5, 6, 7])('keeps well-separated clusters apart and lowers the KL divergence (seed %i)', seed => {
    const random = seededRandom(seed);
    const { X, labels } = clusters(4, 30, 10, random);
    const n = labels.length;
    const highNeighbours = nearestNeighbours(squaredDistances(X, n, 10), n, 5);
    const tsne = new Tsne(affinities(X, n, 10, 20), n, { random });
    const kl: number[] = [];
    for (let it = 0; it < 600; it++) {
      tsne.step();
      kl.push(tsne.kl);
    }
    expect(tsne.exaggerating).toBe(false);
    // After early exaggeration the true KL keeps falling.
    expect(kl[599]).toBeLessThan(kl[260]);
    expect(kl[599]).toBeLessThan(kl[0]);
    expect(knnAgreement(tsne.Y, 2, labels, n, 5)).toBeGreaterThan(0.97);
    expect(neighbourhoodKept(highNeighbours, tsne.Y, 2, n, 5)).toBeGreaterThan(0.3);
    expect(tsne.Y.every(Number.isFinite)).toBe(true);
  });

  it('is deterministic for a seed, and another seed gives another layout', () => {
    const n = 40, P = affinities(randomPoints(n, 5, seededRandom(6)), n, 5, 10);
    const run = (seed: number) => {
      const t = new Tsne(P, n, { random: seededRandom(seed) });
      for (let i = 0; i < 50; i++) t.step();
      return t.Y;
    };
    expect(run(1)).toEqual(run(1));
    expect(run(1)).not.toEqual(run(2));
  });
});
