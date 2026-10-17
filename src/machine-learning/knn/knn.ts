/**
 * k-nearest neighbours: classify a point by asking the training points closest to it.
 *
 * There is no training at all: the model is the training set. To classify x, find the k
 * training points nearest to x (Euclidean distance) and let them vote:
 *   uniform    every neighbour has one vote
 *   distance   neighbour i votes with weight 1/dᵢ, so closer neighbours count more
 * The vote fractions estimate the class probabilities p(c | x), and the predicted class is
 * the one with the most votes (a tie goes to the class of the nearest of the tied neighbours).
 *
 * k is the only knob. k = 1 reproduces every training label exactly, noise included: it
 * memorises. Larger k averages over a wider neighbourhood, which smooths the boundary and
 * ignores isolated noisy points, until k is so large that the neighbourhood spans several
 * classes and the boundary washes out.
 *
 * Distance is only meaningful if it tells near from far. For random points in d dimensions
 * the nearest and the farthest neighbour end up at almost the same distance as d grows (the
 * curse of dimensionality), see `distanceRatio`. kNN still works on images because real
 * images are far from random: they lie close to a low-dimensional surface in pixel space.
 */

export type Weighting = 'uniform' | 'distance';

export interface Neighbours {
  /** Row indices of the k nearest points, nearest first, and their Euclidean distances. */
  index: number[];
  dist: number[];
  /** Distance to the farthest point of the set (for the nearest / farthest ratio). */
  farthest: number;
}

/**
 * The k rows of `points` (n × dim, row-major) nearest to `query` (dim values from `offset`).
 * A brute-force scan that keeps a sorted list of the best k so far.
 */
export function nearest(points: ArrayLike<number>, dim: number, query: ArrayLike<number>, k: number, offset = 0): Neighbours {
  const d2 = points instanceof Float32Array && query instanceof Float32Array
    ? squaredDistancesF32(points, dim, query, offset)
    : squaredDistances(points, dim, query, offset);
  const index: number[] = [], best: number[] = [];
  let far = 0;
  for (let i = 0; i < d2.length; i++) {
    const s = d2[i];
    if (s > far) far = s;
    if (index.length === k && s >= best[k - 1]) continue;
    // Insert i in order (insertion sort into a list of at most k).
    let at = Math.min(index.length, k - 1);
    while (at > 0 && best[at - 1] > s) at--;
    index.splice(at, 0, i);
    best.splice(at, 0, s);
    if (index.length > k) { index.pop(); best.pop(); }
  }
  return { index, dist: best.map(Math.sqrt), farthest: Math.sqrt(far) };
}

/** ‖pointᵢ − query‖² for every row i. */
function squaredDistances(points: ArrayLike<number>, dim: number, query: ArrayLike<number>, offset: number): Float64Array {
  const out = new Float64Array(Math.floor(points.length / dim));
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    for (let c = 0; c < dim; c++) { const t = points[i * dim + c] - query[offset + c]; s += t * t; }
    out[i] = s;
  }
  return out;
}

/** The same for images (Float32Array): a separate copy so the JavaScript engine can specialise this hot loop. */
function squaredDistancesF32(points: Float32Array, dim: number, query: Float32Array, offset: number): Float64Array {
  const out = new Float64Array(Math.floor(points.length / dim));
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    const row = i * dim;
    for (let c = 0; c < dim; c++) { const t = points[row + c] - query[offset + c]; s += t * t; }
    out[i] = s;
  }
  return out;
}

/** Vote fractions of the first k neighbours, one per class (they sum to 1). */
export function vote(nb: Neighbours, labels: ArrayLike<number>, classes: number, k: number, weighting: Weighting): Float64Array {
  const votes = new Float64Array(classes);
  const m = Math.min(k, nb.index.length);
  for (let r = 0; r < m; r++) votes[labels[nb.index[r]]] += weighting === 'uniform' ? 1 : 1 / (nb.dist[r] + 1e-9);
  let total = 0;
  for (const v of votes) total += v;
  if (total > 0) for (let c = 0; c < classes; c++) votes[c] /= total;
  return votes;
}

/** The class with the most votes; a tie goes to the tied class whose neighbour is nearest. */
export function decide(votes: ArrayLike<number>, nb: Neighbours, labels: ArrayLike<number>): number {
  let best = 0;
  for (let c = 1; c < votes.length; c++) if (votes[c] > votes[best]) best = c;
  for (const i of nb.index) if (Math.abs(votes[labels[i]] - votes[best]) < 1e-12) return labels[i];
  return best;
}

export interface LabelledPoints {
  /** n × dim coordinates and n labels in 0 … classes − 1. */
  x: ArrayLike<number>;
  labels: ArrayLike<number>;
  dim: number;
  classes: number;
}

/** Class probabilities (vote fractions) at a query point. */
export function predictProba(train: LabelledPoints, query: ArrayLike<number>, k: number, weighting: Weighting): Float64Array {
  return vote(nearest(train.x, train.dim, query, k), train.labels, train.classes, k, weighting);
}

export function predict(train: LabelledPoints, query: ArrayLike<number>, k: number, weighting: Weighting): number {
  const nb = nearest(train.x, train.dim, query, k);
  return decide(vote(nb, train.labels, train.classes, k, weighting), nb, train.labels);
}

/**
 * Accuracy on `queries` for every k in `ks`, finding the neighbours once (up to the
 * largest k) and re-counting the votes per k.
 */
export function accuracyByK(train: LabelledPoints, queries: LabelledPoints, ks: readonly number[], weighting: Weighting): number[] {
  const kMax = Math.max(...ks);
  const right = new Array<number>(ks.length).fill(0);
  const n = queries.labels.length;
  for (let q = 0; q < n; q++) {
    const nb = nearest(train.x, train.dim, queries.x, kMax, q * queries.dim);
    ks.forEach((k, j) => {
      if (decide(vote(nb, train.labels, train.classes, k, weighting), nb, train.labels) === queries.labels[q]) right[j]++;
    });
  }
  return right.map(r => r / n);
}

/**
 * The curse of dimensionality, measured: for `n` points uniform in the unit cube [0, 1]^d
 * and a uniform query, the mean over `queries` queries of (nearest distance) / (farthest
 * distance). Near 0 in low dimensions, it climbs towards 1 as d grows: every point is
 * then about as far away as every other.
 */
export function distanceRatio(d: number, n: number, queries: number, random: () => number): number {
  const points = Float64Array.from({ length: n * d }, random);
  const query = new Float64Array(d);
  let sum = 0;
  for (let q = 0; q < queries; q++) {
    for (let c = 0; c < d; c++) query[c] = random();
    const nb = nearest(points, d, query, 1);
    sum += nb.dist[0] / nb.farthest;
  }
  return sum / queries;
}

/** Distances from one uniform query to `n` uniform points in [0, 1]^d (for a histogram). */
export function uniformDistances(d: number, n: number, random: () => number): Float64Array {
  const query = Float64Array.from({ length: d }, random);
  return Float64Array.from({ length: n }, () => {
    let s = 0;
    for (let c = 0; c < d; c++) { const t = random() - query[c]; s += t * t; }
    return Math.sqrt(s);
  });
}
