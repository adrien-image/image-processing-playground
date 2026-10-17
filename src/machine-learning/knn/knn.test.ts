import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { makeClassification } from '../shared/datasets';
import { accuracyByK, distanceRatio, nearest, predict, predictProba, uniformDistances, vote, type LabelledPoints } from './knn';

const points2d = (shape: Parameters<typeof makeClassification>[0], n: number, noise: number, seed: number, classes = 2): LabelledPoints => {
  const d = makeClassification(shape, n, noise, seededRandom(seed), classes);
  return { x: d.xy, labels: d.labels, dim: 2, classes: d.classes };
};

describe('nearest', () => {
  it('returns the k closest rows in order, with their distances and the farthest distance', () => {
    const pts = [0, 0, 3, 4, 1, 0, -2, 0, 0, 0.5];
    const nb = nearest(pts, 2, [0, 0], 3);
    expect(nb.index).toEqual([0, 4, 2]);
    expect(nb.dist).toEqual([0, 0.5, 1]);
    expect(nb.farthest).toBe(5);
  });

  it('agrees with sorting all distances', () => {
    const random = seededRandom(3);
    const pts = Float64Array.from({ length: 300 * 5 }, random), q = Float64Array.from({ length: 5 }, random);
    const all = Array.from({ length: 300 }, (_, i) => [i, Math.hypot(...Array.from({ length: 5 }, (_, c) => pts[i * 5 + c] - q[c]))]);
    all.sort((a, b) => a[1] - b[1]);
    const nb = nearest(pts, 5, q, 11);
    expect(nb.index).toEqual(all.slice(0, 11).map(a => a[0]));
  });
});

describe('vote and predict', () => {
  // Two class-1 points close to the query, three class-0 points farther away.
  const train: LabelledPoints = { x: [0.1, 0, -0.1, 0, 1, 0, 0, 1, -1, 0.2], labels: [1, 1, 0, 0, 0], dim: 2, classes: 2 };

  it('uniform voting follows the majority, distance weighting the closest', () => {
    expect(predict(train, [0, 0], 5, 'uniform')).toBe(0);              // 3 votes against 2
    expect(predict(train, [0, 0], 5, 'distance')).toBe(1);             // 2 × 1/0.1 beats 1/1 + 1/1 + 1/1.02
    expect(Array.from(predictProba(train, [0, 0], 5, 'uniform'))).toEqual([0.6, 0.4]);
    const p = predictProba(train, [0, 0], 5, 'distance');
    expect(p[0] + p[1]).toBeCloseTo(1, 12);
    expect(p[1]).toBeCloseTo(20 / (20 + 2 + 1 / Math.hypot(1, 0.2)), 6);
  });

  it('breaks a tie in favour of the nearest neighbour', () => {
    const nb = nearest(train.x, 2, [0.05, 0], 4); // 1, 1, 0, 0: a tie, and the nearest is class 1
    expect(Array.from(vote(nb, train.labels, 2, 4, 'uniform'))).toEqual([0.5, 0.5]);
    expect(predict(train, [0.05, 0], 4, 'uniform')).toBe(1);
    expect(predict(train, [-0.08, 0], 2, 'uniform')).toBe(1);
  });

  it('with k = 1 reproduces every training label, however noisy', () => {
    const data = points2d('moons', 200, 0.4, 1);
    expect(accuracyByK(data, data, [1], 'uniform')).toEqual([1]);
    // ...and with distance weighting each training point outvotes the others at any k.
    expect(accuracyByK(data, data, [51], 'distance')).toEqual([1]);
    expect(accuracyByK(data, data, [51], 'uniform')[0]).toBeLessThan(1);
  });

  it('generalises: on noisy moons a moderate k beats both k = 1 and a very large k on test points', () => {
    const train = points2d('moons', 200, 0.3, 2), test = points2d('moons', 400, 0.3, 3);
    const [k1, k15] = accuracyByK(train, test, [1, 15], 'uniform');
    expect(k15).toBeGreaterThan(k1);
    expect(k15).toBeGreaterThan(0.85);
    const [k151] = accuracyByK(train, test, [151], 'uniform');
    expect(k151).toBeLessThan(k15);
  });

  it('handles three classes', () => {
    const train = points2d('blobs', 150, 0, 4, 3), test = points2d('blobs', 300, 0, 5, 3);
    expect(accuracyByK(train, test, [5], 'uniform')[0]).toBeGreaterThan(0.95);
  });
});

describe('distanceRatio', () => {
  it('climbs towards 1 as the dimension grows', () => {
    const random = seededRandom(7);
    const ratios = [1, 2, 10, 100, 1000].map(d => distanceRatio(d, 300, 10, random));
    for (let i = 1; i < ratios.length; i++) expect(ratios[i]).toBeGreaterThan(ratios[i - 1]);
    expect(ratios[0]).toBeLessThan(0.02);
    expect(ratios[4]).toBeGreaterThan(0.85);
  });

  it('concentrates the distances around √(d/6)', () => {
    const dist = uniformDistances(500, 1000, seededRandom(8));
    const mean = dist.reduce((a, b) => a + b, 0) / dist.length;
    expect(mean).toBeGreaterThan(0.95 * Math.sqrt(500 / 6));
    expect(mean).toBeLessThan(1.05 * Math.sqrt(500 / 6));
  });
});
