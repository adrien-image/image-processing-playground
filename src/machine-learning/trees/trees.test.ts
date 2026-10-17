import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import { makeClassification, subset } from '../shared/datasets';
import {
  accuracy, accuracyAtDepths, argmax, decisionPath, fitClassificationTree, fitForest, fitRegressionTree, forestProba, impurityOf,
  predictClass, predictProba, predictValue, splitRows, treeSize,
} from './trees';

describe('impurity', () => {
  it('is zero for a pure node and largest for an even mix', () => {
    expect(impurityOf([5, 0], 5, 'gini')).toBe(0);
    expect(impurityOf([0, 5], 5, 'entropy')).toBe(0);
    expect(impurityOf([3, 3], 6, 'gini')).toBeCloseTo(0.5, 12);
    expect(impurityOf([3, 3], 6, 'entropy')).toBeCloseTo(1, 12);
    expect(impurityOf([2, 2, 2], 6, 'entropy')).toBeCloseTo(Math.log2(3), 12);
  });
});

describe('classification tree', () => {
  it('stops at a pure node', () => {
    const tree = fitClassificationTree([0, 0, 1, 1, 2, 2, 3, 3], 2, [1, 1, 1, 1], 2, { maxDepth: 10, minLeaf: 1 });
    expect(tree.split).toBeUndefined();
    expect(tree.impurity).toBe(0);
    expect(tree.value).toBe(1);
  });

  it('finds the split with the largest impurity decrease on hand-made data', () => {
    // Feature 0 separates the classes perfectly between 0.3 and 0.6; feature 1 is noise.
    const X = [0.1, 0.9, 0.2, 0.1, 0.3, 0.5, 0.6, 0.8, 0.7, 0.2, 0.9, 0.4];
    const y = [0, 0, 0, 1, 1, 1];
    const tree = fitClassificationTree(X, 2, y, 2, { maxDepth: 1, minLeaf: 1 });
    expect(tree.split?.feature).toBe(0);
    expect(tree.split?.threshold).toBeCloseTo(0.45, 6);
    expect(tree.split?.gain).toBeCloseTo(0.5, 12); // Gini 0.5 → two pure children
    // Every other threshold on either feature decreases the impurity less.
    for (let f = 0; f < 2; f++) for (let t = -0.05; t < 1; t += 0.05) {
      const left = y.filter((_, i) => X[2 * i + f] <= t), right = y.filter((_, i) => X[2 * i + f] > t);
      const counts = (ys: number[]) => [ys.filter(v => v === 0).length, ys.filter(v => v === 1).length];
      const gain = 0.5 - (left.length / 6) * impurityOf(counts(left), left.length, 'gini') - (right.length / 6) * impurityOf(counts(right), right.length, 'gini');
      expect(gain).toBeLessThanOrEqual(tree.split!.gain + 1e-12);
    }
  });

  it('respects the depth limit and min samples per leaf, and cutting a deep tree equals growing a shallow one', () => {
    const data = makeClassification('spiral', 300, 0.05, seededRandom(4), 3);
    const deep = fitClassificationTree(data.xy, 2, data.labels, 3, { maxDepth: 12, minLeaf: 3 });
    const minLeafOk = (node: typeof deep): boolean => (node.split ? minLeafOk(node.split.left) && minLeafOk(node.split.right) : node.n >= 3);
    expect(minLeafOk(deep)).toBe(true);
    for (const depth of [1, 3, 5]) {
      const shallow = fitClassificationTree(data.xy, 2, data.labels, 3, { maxDepth: depth, minLeaf: 3 });
      expect(treeSize(shallow).depth).toBe(depth);
      for (let i = 0; i < data.n; i++) {
        const x = [data.xy[2 * i], data.xy[2 * i + 1]];
        expect(predictProba(shallow, x)).toEqual(predictProba(deep, x, depth));
      }
    }
    // Unlimited depth with one point per leaf memorises the training set.
    const full = fitClassificationTree(data.xy, 2, data.labels, 3, { maxDepth: 30, minLeaf: 1 });
    expect(accuracy(x => predictClass(full, x), data.xy, 2, data.labels)).toBe(1);
  });

  it('records the path of a point from the root to its leaf', () => {
    const data = makeClassification('moons', 200, 0.1, seededRandom(2));
    const tree = fitClassificationTree(data.xy, 2, data.labels, 2, { maxDepth: 4, minLeaf: 1 });
    const path = decisionPath(tree, [0.2, 0.3]);
    expect(path[0]).toBe(tree);
    path.forEach((node, k) => expect(node.depth).toBe(k));
    expect(path[path.length - 1].split).toBeUndefined();
    expect(path[path.length - 1].value).toBe(predictClass(tree, [0.2, 0.3]));
  });
});

describe('random forest', () => {
  it('with one tree, no bootstrap and every feature at each split is a single tree', () => {
    const data = makeClassification('xor', 200, 0.1, seededRandom(5));
    const opts = { maxDepth: 6, minLeaf: 2, impurity: 'entropy' as const };
    const tree = fitClassificationTree(data.xy, 2, data.labels, 2, opts);
    const forest = fitForest(data.xy, 2, data.labels, 2, { ...opts, trees: 1, bootstrap: false, maxFeatures: 2 });
    for (let k = 0; k < 400; k++) {
      const x = [Math.sin(k) * 1.4, Math.cos(1.7 * k) * 1.4];
      expect(forestProba(forest, x)).toEqual(predictProba(tree, x));
    }
  });

  it('bootstraps about 63% of the distinct points per tree', () => {
    const data = makeClassification('blobs', 400, 0.1, seededRandom(1), 3);
    const forest = fitForest(data.xy, 2, data.labels, 3, { maxDepth: 3, minLeaf: 1, trees: 20, bootstrap: true, maxFeatures: 1 });
    const share = forest.inBag.reduce((s, bag) => s + bag.filter(c => c > 0).length / bag.length, 0) / 20;
    expect(share).toBeGreaterThan(0.6);
    expect(share).toBeLessThan(0.66);
  });

  it('beats a deep tree on noisy test data', () => {
    let margin = 0;
    for (const seed of [1, 2, 3]) {
      const all = makeClassification('moons', 600, 0.25, seededRandom(seed));
      const rows = splitRows(600, 0.5, seededRandom(seed + 10));
      const train = subset(all, rows.train), test = subset(all, rows.test);
      const opts = { maxDepth: 12, minLeaf: 1 };
      const tree = fitClassificationTree(train.xy, 2, train.labels, 2, opts);
      const forest = fitForest(train.xy, 2, train.labels, 2, { ...opts, trees: 60, bootstrap: true, maxFeatures: 1 }, seed);
      const treeTest = accuracy(x => predictClass(tree, x), test.xy, 2, test.labels);
      const forestTest = accuracy(x => argmax(forestProba(forest, x)), test.xy, 2, test.labels);
      expect(accuracy(x => predictClass(tree, x), train.xy, 2, train.labels)).toBe(1); // the deep tree memorises
      expect(forestTest).toBeGreaterThanOrEqual(treeTest);
      margin += (forestTest - treeTest) / 3;
      // The per-depth curves agree with predicting at each depth one by one.
      const curve = accuracyAtDepths(forest.trees, test.xy, 2, test.labels, 2, 12);
      expect(curve[11]).toBeCloseTo(forestTest, 12);
      expect(accuracyAtDepths([tree], test.xy, 2, test.labels, 2, 12)[2]).toBeCloseTo(accuracy(x => predictClass(tree, x, 3), test.xy, 2, test.labels), 12);
    }
    expect(margin).toBeGreaterThan(0.02);
  });
});

describe('regression tree', () => {
  it('fits piecewise-constant means and splits where the step is', () => {
    const x = [0, 1, 2, 3, 4, 5], y = [1, 1, 1, 5, 5, 5];
    const tree = fitRegressionTree(x, 1, y, { maxDepth: 3, minLeaf: 1 });
    expect(tree.value).toBeCloseTo(3, 12);
    expect(tree.impurity).toBeCloseTo(4, 12);
    expect(tree.split?.threshold).toBeCloseTo(2.5, 12);
    expect(tree.split?.gain).toBeCloseTo(4, 12);
    expect(tree.split?.left.split).toBeUndefined(); // pure in y: nothing left to explain
    expect(predictValue(tree, [0.5])).toBe(1);
    expect(predictValue(tree, [4.2])).toBe(5);
  });

  it('respects minLeaf and is the mean of its leaf', () => {
    const random = seededRandom(3);
    const x = Array.from({ length: 50 }, () => random()), y = x.map(v => Math.sin(6 * v) + 0.1 * random());
    const tree = fitRegressionTree(x, 1, y, { maxDepth: 4, minLeaf: 5 });
    const leaves: number[] = [];
    const walk = (n: typeof tree): void => { if (n.split) { walk(n.split.left); walk(n.split.right); } else leaves.push(n.n); };
    walk(tree);
    expect(Math.min(...leaves)).toBeGreaterThanOrEqual(5);
    expect(leaves.reduce((a, b) => a + b, 0)).toBe(50);
  });
});
