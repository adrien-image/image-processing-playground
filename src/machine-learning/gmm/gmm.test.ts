import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import {
  agreement, assignNearest, bicCurve, eStep, fitGmm, hardLabels, inertia, initialCentres, makeMixtureData, mStep, objective,
  sampleMixture, startMixture, trueMixture, updateCentres, type Mixture,
} from './gmm';

/** The best of a few k-means++ starts, as one would do in practice. */
function bestFit(xy: Float32Array, K: number, seed: number) {
  const random = seededRandom(seed);
  let best: ReturnType<typeof fitGmm> | null = null;
  for (let r = 0; r < 4; r++) {
    const fit = fitGmm(xy, startMixture(xy, initialCentres(xy, K, random)), 300);
    if (!best || fit.history.at(-1)! > best.history.at(-1)!) best = fit;
  }
  return best!;
}

describe('EM', () => {
  it('never decreases the (penalised) log-likelihood, from good and bad starts', () => {
    for (const kind of ['stretched', 'unequal', 'round'] as const) {
      for (const method of ['kmeans++', 'random'] as const) {
        const { xy } = makeMixtureData(kind, 300, seededRandom(1));
        for (const K of [2, 3, 5]) {
          const { history } = fitGmm(xy, startMixture(xy, initialCentres(xy, K, seededRandom(K), method)), 200);
          for (let t = 1; t < history.length; t++) expect(history[t]).toBeGreaterThanOrEqual(history[t - 1] - 1e-9 * Math.abs(history[t - 1]));
          expect(history.at(-1)!).toBeGreaterThan(history[0]);
        }
      }
    }
  });

  it('recovers the parameters of well-separated clusters, up to relabelling', () => {
    const truth: Mixture = [
      { weight: 0.5, mean: [-0.6, 0.4], cov: [0.03, 0.01, 0.02] },
      { weight: 0.3, mean: [0.6, 0.5], cov: [0.015, -0.005, 0.03] },
      { weight: 0.2, mean: [0.1, -0.7], cov: [0.02, 0, 0.01] },
    ];
    const { xy } = sampleMixture(truth, 2000, seededRandom(4));
    const { mix } = bestFit(xy, 3, 5);
    for (const t of truth) {
      const c = mix.reduce((a, b) => (Math.hypot(b.mean[0] - t.mean[0], b.mean[1] - t.mean[1]) < Math.hypot(a.mean[0] - t.mean[0], a.mean[1] - t.mean[1]) ? b : a));
      expect(Math.hypot(c.mean[0] - t.mean[0], c.mean[1] - t.mean[1])).toBeLessThan(0.02);
      expect(Math.abs(c.weight - t.weight)).toBeLessThan(0.03);
      for (let j = 0; j < 3; j++) expect(Math.abs(c.cov[j] - t.cov[j])).toBeLessThan(0.004);
    }
  });

  it('gives responsibilities that sum to 1 for every point', () => {
    const { xy } = makeMixtureData('unequal', 300, seededRandom(2));
    const K = 4;
    let mix = startMixture(xy, initialCentres(xy, K, seededRandom(3)));
    for (let t = 0; t < 20; t++) {
      const { resp } = eStep(xy, mix);
      for (let i = 0; i < 300; i++) {
        let s = 0;
        for (let k = 0; k < K; k++) { expect(resp[i * K + k]).toBeGreaterThanOrEqual(0); s += resp[i * K + k]; }
        expect(s).toBeCloseTo(1, 12);
      }
      mix = mStep(xy, resp, K);
    }
  });

  it('keeps every covariance positive definite, even for a component sitting on one isolated point', () => {
    // 200 points in a blob and one far outlier, with a centre placed exactly on it.
    const blob = sampleMixture([{ weight: 1, mean: [0, 0], cov: [0.04, 0, 0.04] }], 200, seededRandom(6)).xy;
    const xy = Float32Array.from([...blob, 1.3, 1.3]);
    let mix = startMixture(xy, [[0, 0], [0.1, -0.1], [1.3, 1.3]]);
    for (let t = 0; t < 100; t++) {
      mix = mStep(xy, eStep(xy, mix).resp, 3);
      for (const { cov: [a, b, c] } of mix) {
        expect(a).toBeGreaterThan(0);
        expect(a * c - b * b).toBeGreaterThan(0);
      }
      expect(Number.isFinite(objective(xy, mix))).toBe(true);
    }
    // The lonely component keeps a variance of about λ = 0.001 instead of collapsing to 0.
    const lonely = mix.find(c => c.mean[0] > 1)!;
    expect(lonely.cov[0]).toBeGreaterThan(5e-4);
  });
});

describe('k-means and GMM on the same starts', () => {
  it("Lloyd's inertia never increases", () => {
    const { xy } = makeMixtureData('stretched', 300, seededRandom(1));
    let centres = initialCentres(xy, 3, seededRandom(1));
    let prev = Infinity;
    for (let t = 0; t < 30; t++) {
      const labels = assignNearest(xy, centres);
      const j = inertia(xy, labels, centres);
      expect(j).toBeLessThanOrEqual(prev + 1e-9);
      prev = j;
      centres = updateCentres(xy, labels, centres);
    }
  });

  it('agree on round blobs, while only the GMM finds stretched and unequal clusters', () => {
    for (const kind of ['round', 'stretched', 'unequal'] as const) {
      const data = makeMixtureData(kind, 300, seededRandom(3));
      const { mix } = bestFit(data.xy, 3, 7);
      const gmm = agreement(hardLabels(eStep(data.xy, mix).resp, 3), data.labels, 3);
      let centres = initialCentres(data.xy, 3, seededRandom(7));
      for (let t = 0; t < 50; t++) centres = updateCentres(data.xy, assignNearest(data.xy, centres), centres);
      const km = agreement(assignNearest(data.xy, centres), data.labels, 3);
      expect(gmm).toBeGreaterThan(0.97);
      expect(km > 0.97).toBe(kind === 'round');
    }
  });

  it('BIC picks the true number of components', () => {
    for (const kind of ['round', 'stretched', 'unequal'] as const) {
      const { xy } = makeMixtureData(kind, 300, seededRandom(5));
      const curve = bicCurve(xy, 6, seededRandom(5));
      expect(curve.indexOf(Math.min(...curve)) + 1).toBe(3);
    }
  });

  it('agreement is invariant to relabelling', () => {
    const truth = [0, 0, 1, 1, 2, 2];
    expect(agreement([2, 2, 0, 0, 1, 1], truth, 3)).toBe(1);
    expect(agreement([0, 0, 0, 1, 2, 2], truth, 3)).toBeCloseTo(5 / 6, 12);
    expect(trueMixture('round')).toHaveLength(3);
  });
});
