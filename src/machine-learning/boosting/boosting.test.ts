import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { makeRegression } from '../shared/datasets';
import { splitRows } from '../trees/trees';
import { boostRound, gradientBoost, mse, predictBoosted, startBoosting } from './boosting';

function data(seed: number, nTrain = 80, noise = 0.3) {
  const d = makeRegression(nTrain + 300, noise, seededRandom(seed));
  const rows = splitRows(d.n, 300 / d.n, seededRandom(seed + 50));
  const pick = (r: number[]) => ({ x: r.map(i => d.x[i]), y: r.map(i => d.y[i]) });
  return { train: pick(rows.train), test: pick(rows.test) };
}

describe('gradient boosting', () => {
  it('starts from F₀ = mean(y)', () => {
    const { train, test } = data(1);
    const s = startBoosting(train, test, [0]);
    const mean = train.y.reduce((a, b) => a + b, 0) / train.y.length;
    expect(s.f0).toBeCloseTo(mean, 12);
    expect(predictBoosted(s, 0.3)).toBeCloseTo(mean, 12);
    expect(s.trainMse[0]).toBeCloseTo(mse(train.y, train.y.map(() => mean)), 12);
  });

  it('fits each tree to the residuals and adds η times it', () => {
    const { train, test } = data(2);
    const grid = [-0.5, 0, 0.5];
    const s0 = startBoosting(train, test, grid);
    const s1 = boostRound(s0, train, test, grid, { depth: 2, rate: 0.5, minLeaf: 1 });
    s1.residuals.forEach((r, i) => expect(r).toBeCloseTo(train.y[i] - s0.f0, 12));
    grid.forEach((x, i) => expect(s1.gridF[i]).toBeCloseTo(predictBoosted(s1, x), 12));
    expect(s0.rounds).toHaveLength(0); // the old state is left as it was
  });

  it('drives the training error to ~0 with learning rate 1 and deep trees', () => {
    const { train, test } = data(3);
    const s = gradientBoost(train, test, { depth: 12, rate: 1, minLeaf: 1 }, 3);
    expect(s.trainMse[3]).toBeLessThan(1e-20);
  });

  it('never increases the training error with a small learning rate', () => {
    for (const seed of [1, 2, 3]) {
      const { train, test } = data(seed);
      for (const depth of [1, 2, 4]) {
        const s = gradientBoost(train, test, { depth, rate: 0.1, minLeaf: 1 }, 200);
        for (let m = 1; m <= 200; m++) expect(s.trainMse[m]).toBeLessThanOrEqual(s.trainMse[m - 1] + 1e-12);
        expect(s.trainMse[200]).toBeLessThan(s.trainMse[0] / 4);
      }
    }
  });

  it('overfits with deep trees, a high rate and many rounds; a small rate generalises better', () => {
    for (const seed of [1, 2, 3, 4]) {
      const { train, test } = data(seed);
      const fast = gradientBoost(train, test, { depth: 3, rate: 0.3, minLeaf: 1 }, 300);
      const best = Math.min(...fast.testMse), at = fast.testMse.indexOf(best);
      expect(at).toBeLessThan(60);
      expect(fast.testMse[300]).toBeGreaterThan(1.2 * best);
      expect(fast.trainMse[300]).toBeLessThan(1e-6);
      const slow = gradientBoost(train, test, { depth: 1, rate: 0.05, minLeaf: 1 }, 300);
      expect(slow.testMse.indexOf(Math.min(...slow.testMse))).toBeGreaterThan(at); // needs more rounds…
      expect(slow.testMse[300]).toBeLessThan(fast.testMse[300]); // …but ends better
    }
  });
});
