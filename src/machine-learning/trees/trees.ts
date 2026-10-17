/**
 * Decision trees (CART) and random forests.
 *
 * A tree asks one yes/no question per node, "is feature f ≤ threshold t?", and sends
 * the point left or right until it reaches a leaf, which answers with the classes (or
 * the mean value) of the training points that ended up there.
 *
 * Training is greedy and top-down. At each node we try every feature and every
 * threshold halfway between two consecutive distinct values, and keep the split that
 * lowers the impurity the most:
 *   gain = I(parent) − (n_L / n)·I(left) − (n_R / n)·I(right)
 *   Gini      I = 1 − Σ_c p_c²          (chance that two random points disagree)
 *   entropy   I = −Σ_c p_c · log₂ p_c   (bits needed to name the class)
 *   variance  I = mean((y − ȳ)²)        (for regression trees)
 * A node becomes a leaf when it is pure, when it reaches maxDepth, when it has fewer
 * than 2·minLeaf points, or when no split improves it.
 *
 * Because growth is greedy and top-down, the tree grown with maxDepth d is exactly the
 * tree grown deeper and cut at depth d: `predict…(node, x, d)` does that cut, so one
 * deep tree answers for every depth.
 *
 * A random forest grows many trees, each on a bootstrap sample (n points drawn with
 * replacement) and looking at a random subset of the features at each split, and
 * averages their class proportions. Points are rows of a row-major array X (n × d).
 */
import { seededRandom } from '../../lib/random';

export type Impurity = 'gini' | 'entropy';

export interface TreeNode {
  depth: number;
  /** Number of training points that reached the node (bootstrap copies counted). */
  n: number;
  impurity: number;
  /** Classification: points of each class at this node. Regression: empty. */
  counts: number[];
  /** Classification: the majority class. Regression: the mean of y (the prediction). */
  value: number;
  /** Absent on a leaf. Points with x[feature] ≤ threshold go left. */
  split?: { feature: number; threshold: number; gain: number; left: TreeNode; right: TreeNode };
}

export interface TreeOptions {
  maxDepth: number;
  minLeaf: number;
  /** Classification only. */
  impurity?: Impurity;
  /** Features looked at per split (random forests use fewer than d). Default: all of them. */
  maxFeatures?: number;
  /** Seed for the random choice of features; each node draws from its own stream, so cutting a tree changes nothing above the cut. */
  seed?: number;
}

/** Impurity of a node from its class counts. */
export function impurityOf(counts: ArrayLike<number>, n: number, kind: Impurity): number {
  if (n === 0) return 0;
  let s = 0;
  for (let c = 0; c < counts.length; c++) {
    const p = counts[c] / n;
    if (kind === 'gini') s += p * p;
    else if (p > 0) s -= p * Math.log2(p);
  }
  return kind === 'gini' ? 1 - s : s;
}

/**
 * The features to look at, in a random order. With maxFeatures < d a split searches
 * the first maxFeatures of them, and only goes on to the others if none of those can
 * split the node (as scikit-learn does).
 */
function featureOrder(d: number, nodeId: number, seed: number | undefined): number[] {
  const order = Array.from({ length: d }, (_, f) => f);
  if (seed === undefined) return order;
  const random = seededRandom((seed * 2654435761 + nodeId * 40503) >>> 0);
  for (let i = d - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

interface Best { feature: number; threshold: number; gain: number; left: number[]; right: number[] }

/**
 * The best split of the points `rows` on one feature: sort them by that feature, then
 * sweep the threshold from left to right, moving one point at a time from the right
 * child's statistics to the left child's. `score(i)` returns the gain when the first
 * i + 1 sorted points go left.
 */
function sweep(X: ArrayLike<number>, d: number, rows: number[], feature: number, minLeaf: number,
  move: (row: number) => void, score: (nLeft: number) => number, best: Best | null): Best | null {
  const sorted = [...rows].sort((a, b) => X[a * d + feature] - X[b * d + feature]);
  const m = sorted.length;
  let found = best;
  for (let i = 0; i < m - 1; i++) {
    move(sorted[i]);
    const v = X[sorted[i] * d + feature], w = X[sorted[i + 1] * d + feature];
    const nLeft = i + 1;
    if (v === w || nLeft < minLeaf || m - nLeft < minLeaf) continue;
    const gain = score(nLeft);
    // Ties go to the lower feature index, so the order features were tried in never matters.
    const better = !found ? gain > 1e-12 : gain > found.gain + 1e-12 || (gain > found.gain - 1e-12 && feature < found.feature);
    if (better) {
      found = { feature, threshold: (v + w) / 2, gain, left: sorted.slice(0, nLeft), right: sorted.slice(nLeft) };
    }
  }
  return found;
}

/**
 * Grow a classification tree on the points `rows` (indices into X and labels; repeats
 * allowed, as in a bootstrap sample).
 */
export function fitClassificationTree(X: ArrayLike<number>, d: number, labels: ArrayLike<number>, classes: number,
  options: TreeOptions, rows: number[] = Array.from({ length: labels.length }, (_, i) => i)): TreeNode {
  const kind = options.impurity ?? 'gini';
  const grow = (rows: number[], depth: number, nodeId: number): TreeNode => {
    const counts = new Array<number>(classes).fill(0);
    for (const r of rows) counts[labels[r]]++;
    const n = rows.length, impurity = impurityOf(counts, n, kind);
    const node: TreeNode = { depth, n, impurity, counts, value: counts.indexOf(Math.max(...counts)) };
    if (impurity === 0 || depth >= options.maxDepth || n < 2 * options.minLeaf) return node;

    let best: Best | null = null;
    const order = featureOrder(d, nodeId, options.seed);
    for (const [k, feature] of order.entries()) {
      if (best && k >= (options.maxFeatures ?? d)) break;
      const left = new Array<number>(classes).fill(0), right = [...counts];
      best = sweep(X, d, rows, feature, options.minLeaf,
        r => { left[labels[r]]++; right[labels[r]]--; },
        nLeft => impurity - (nLeft / n) * impurityOf(left, nLeft, kind) - ((n - nLeft) / n) * impurityOf(right, n - nLeft, kind),
        best);
    }
    if (!best) return node;
    const { feature, threshold, gain, left, right } = best;
    node.split = { feature, threshold, gain, left: grow(left, depth + 1, 2 * nodeId), right: grow(right, depth + 1, 2 * nodeId + 1) };
    return node;
  };
  return grow(rows, 0, 1);
}

/** Grow a regression tree (squared error) on targets y; the boosting lesson fits these to residuals. */
export function fitRegressionTree(X: ArrayLike<number>, d: number, y: ArrayLike<number>, options: TreeOptions,
  rows: number[] = Array.from({ length: y.length }, (_, i) => i)): TreeNode {
  const grow = (rows: number[], depth: number, nodeId: number): TreeNode => {
    const n = rows.length;
    let sum = 0, sumSq = 0;
    for (const r of rows) { sum += y[r]; sumSq += y[r] * y[r]; }
    const mean = sum / n, impurity = Math.max(0, sumSq / n - mean * mean);
    const node: TreeNode = { depth, n, impurity, counts: [], value: mean };
    if (impurity < 1e-14 || depth >= options.maxDepth || n < 2 * options.minLeaf) return node;

    let best: Best | null = null;
    const order = featureOrder(d, nodeId, options.seed);
    for (const [k, feature] of order.entries()) {
      if (best && k >= (options.maxFeatures ?? d)) break;
      let lSum = 0, lSq = 0;
      best = sweep(X, d, rows, feature, options.minLeaf,
        r => { lSum += y[r]; lSq += y[r] * y[r]; },
        nLeft => {
          const nRight = n - nLeft, rSum = sum - lSum, rSq = sumSq - lSq;
          // n · variance = Σy² − (Σy)² / n, for each child.
          return impurity - (lSq - (lSum * lSum) / nLeft + rSq - (rSum * rSum) / nRight) / n;
        },
        best);
    }
    if (!best) return node;
    const { feature, threshold, gain, left, right } = best;
    node.split = { feature, threshold, gain, left: grow(left, depth + 1, 2 * nodeId), right: grow(right, depth + 1, 2 * nodeId + 1) };
    return node;
  };
  return grow(rows, 0, 1);
}

/** The nodes a point visits from the root to its leaf (or to depth maxDepth). */
export function decisionPath(node: TreeNode, x: ArrayLike<number>, maxDepth = Infinity): TreeNode[] {
  const path = [node];
  let at = node;
  while (at.split && at.depth < maxDepth) {
    at = x[at.split.feature] <= at.split.threshold ? at.split.left : at.split.right;
    path.push(at);
  }
  return path;
}

const leafOf = (node: TreeNode, x: ArrayLike<number>, maxDepth: number): TreeNode => {
  let at = node;
  while (at.split && at.depth < maxDepth) at = x[at.split.feature] <= at.split.threshold ? at.split.left : at.split.right;
  return at;
};

/** Class proportions of the leaf a point lands in. */
export function predictProba(node: TreeNode, x: ArrayLike<number>, maxDepth = Infinity): number[] {
  const leaf = leafOf(node, x, maxDepth);
  return leaf.counts.map(c => c / leaf.n);
}

export const predictClass = (node: TreeNode, x: ArrayLike<number>, maxDepth = Infinity): number => leafOf(node, x, maxDepth).value;

/** A regression tree's prediction: the mean target of the leaf. */
export const predictValue = (node: TreeNode, x: ArrayLike<number>, maxDepth = Infinity): number => leafOf(node, x, maxDepth).value;

/** Depth of the deepest leaf, and the number of leaves. */
export function treeSize(node: TreeNode, maxDepth = Infinity): { depth: number; leaves: number } {
  if (!node.split || node.depth >= maxDepth) return { depth: node.depth, leaves: 1 };
  const a = treeSize(node.split.left, maxDepth), b = treeSize(node.split.right, maxDepth);
  return { depth: Math.max(a.depth, b.depth), leaves: a.leaves + b.leaves };
}

export interface Forest {
  trees: TreeNode[];
  /** inBag[t][i]: how many times training point i was drawn for tree t. */
  inBag: Uint8Array[];
  classes: number;
}

export interface ForestOptions extends TreeOptions {
  trees: number;
  bootstrap: boolean;
}

/** Grow `trees` classification trees, each on its own bootstrap sample and feature draws. */
export function fitForest(X: ArrayLike<number>, d: number, labels: ArrayLike<number>, classes: number,
  options: ForestOptions, seed = 1): Forest {
  const n = labels.length;
  const trees: TreeNode[] = [], inBag: Uint8Array[] = [];
  for (let t = 0; t < options.trees; t++) {
    const random = seededRandom(seed * 7919 + t);
    const rows = options.bootstrap ? Array.from({ length: n }, () => Math.floor(random() * n)) : Array.from({ length: n }, (_, i) => i);
    const bag = new Uint8Array(n);
    for (const r of rows) bag[r] = Math.min(255, bag[r] + 1);
    trees.push(fitClassificationTree(X, d, labels, classes, { ...options, seed: seed * 104729 + t }, rows));
    inBag.push(bag);
  }
  return { trees, inBag, classes };
}

/** The forest's vote: the average of its trees' class proportions at x (each tree cut at maxDepth). */
export function forestProba(forest: Forest, x: ArrayLike<number>, maxDepth = Infinity, upto = forest.trees.length): number[] {
  const p = new Array<number>(forest.classes).fill(0);
  for (let t = 0; t < upto; t++) {
    const leaf = leafOf(forest.trees[t], x, maxDepth);
    for (let c = 0; c < forest.classes; c++) p[c] += leaf.counts[c] / leaf.n / upto;
  }
  return p;
}

export const argmax = (p: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return best;
};

/**
 * Accuracy of the averaged vote of `trees`, each cut at depth 1, 2, …, maxDepth, in a
 * single walk per point and tree (pass one tree for a single tree's curve).
 */
export function accuracyAtDepths(trees: TreeNode[], X: ArrayLike<number>, d: number, labels: ArrayLike<number>, classes: number, maxDepth: number): number[] {
  const right = new Array<number>(maxDepth).fill(0);
  const votes = new Float64Array(maxDepth * classes);
  const x = new Array<number>(d);
  for (let i = 0; i < labels.length; i++) {
    for (let f = 0; f < d; f++) x[f] = X[i * d + f];
    votes.fill(0);
    for (const tree of trees) {
      let node = tree;
      for (let depth = 1; depth <= maxDepth; depth++) {
        if (node.split && node.depth < depth) node = x[node.split.feature] <= node.split.threshold ? node.split.left : node.split.right;
        for (let c = 0; c < classes; c++) votes[(depth - 1) * classes + c] += node.counts[c] / node.n;
      }
    }
    for (let k = 0; k < maxDepth; k++) if (argmax(votes.subarray(k * classes, (k + 1) * classes)) === labels[i]) right[k]++;
  }
  return right.map(r => (labels.length ? r / labels.length : 0));
}

/** Accuracy of `predict` on the rows of X (row-major, d columns). */
export function accuracy(predict: (x: ArrayLike<number>) => number, X: ArrayLike<number>, d: number, labels: ArrayLike<number>): number {
  let right = 0;
  for (let i = 0; i < labels.length; i++) {
    const x = Array.from({ length: d }, (_, f) => X[i * d + f]);
    if (predict(x) === labels[i]) right++;
  }
  return labels.length ? right / labels.length : 0;
}

/** A random train / test split of the indices 0 … n − 1 (Fisher–Yates shuffle, then cut). */
export function splitRows(n: number, testFraction: number, random: () => number): { train: number[]; test: number[] } {
  const rows = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  const cut = Math.round(n * (1 - testFraction));
  return { train: rows.slice(0, cut), test: rows.slice(cut) };
}
