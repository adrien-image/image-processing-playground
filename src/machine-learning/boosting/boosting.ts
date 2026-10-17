/**
 * Gradient boosting for regression with the squared loss.
 *
 * The model is a sum of small regression trees, built one at a time:
 *   F₀(x) = mean(y)
 *   r_i   = y_i − F_{m−1}(x_i)           the residuals
 *   h_m   = a depth-limited regression tree fitted to (x_i, r_i)
 *   F_m   = F_{m−1} + η · h_m           η: the learning rate, or shrinkage
 * For the loss L = ½ Σ (y_i − F(x_i))², the residual is exactly the negative gradient
 * −∂L/∂F(x_i), so each round is a step of gradient descent in function space: the tree
 * is the best piecewise-constant approximation of that gradient, and η is the step size.
 *
 * Each leaf of h_m predicts the mean residual of its points, so adding η·h_m (0 < η < 2)
 * can only lower the training error: a leaf with n points and mean residual r̄ lowers
 * the sum of squared errors by n·r̄²·η·(2 − η). The test error is another matter.
 */
import { fitRegressionTree, predictValue, type TreeNode } from '../trees/trees';

export interface Data1D {
  x: ArrayLike<number>;
  y: ArrayLike<number>;
}

export interface BoostOptions {
  /** Depth of each tree (1 = a stump: one split, two leaves). */
  depth: number;
  /** Learning rate η, the shrinkage applied to each tree. */
  rate: number;
  minLeaf: number;
}

export interface Round {
  tree: TreeNode;
  rate: number;
}

export interface BoostState {
  f0: number;
  rounds: Round[];
  /** Current predictions F_m on the training points, the test points and the plot grid. */
  trainF: Float64Array;
  testF: Float64Array;
  gridF: Float64Array;
  /** MSE after each round (index 0: F₀ alone). */
  trainMse: number[];
  testMse: number[];
  /** The residuals the newest tree was fitted to (before it was added). */
  residuals: Float64Array;
}

export function mse(y: ArrayLike<number>, f: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += (y[i] - f[i]) ** 2;
  return y.length ? s / y.length : 0;
}

const mean = (v: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  return v.length ? s / v.length : 0;
};

/** F₀: the constant that minimises the squared error, the mean of y. */
export function startBoosting(train: Data1D, test: Data1D, grid: ArrayLike<number>): BoostState {
  const f0 = mean(train.y);
  const trainF = new Float64Array(train.y.length).fill(f0);
  const testF = new Float64Array(test.y.length).fill(f0);
  return {
    f0, rounds: [], trainF, testF, gridF: new Float64Array(grid.length).fill(f0),
    trainMse: [mse(train.y, trainF)], testMse: [mse(test.y, testF)], residuals: new Float64Array(train.y.length),
  };
}

/** One round: fit a tree to the residuals and add η times it to every prediction (a new state; the old one is unchanged). */
export function boostRound(s: BoostState, train: Data1D, test: Data1D, grid: ArrayLike<number>, o: BoostOptions): BoostState {
  const residuals = Float64Array.from(train.y, (y, i) => y - s.trainF[i]);
  const tree = fitRegressionTree(train.x, 1, residuals, { maxDepth: o.depth, minLeaf: o.minLeaf });
  const add = (xs: ArrayLike<number>, f: Float64Array) => Float64Array.from(f, (v, i) => v + o.rate * predictValue(tree, [xs[i]]));
  const trainF = add(train.x, s.trainF), testF = add(test.x, s.testF), gridF = add(grid, s.gridF);
  return {
    f0: s.f0, rounds: [...s.rounds, { tree, rate: o.rate }], trainF, testF, gridF,
    trainMse: [...s.trainMse, mse(train.y, trainF)], testMse: [...s.testMse, mse(test.y, testF)], residuals,
  };
}

/** `rounds` rounds of boosting from scratch. */
export function gradientBoost(train: Data1D, test: Data1D, o: BoostOptions, rounds: number, grid: ArrayLike<number> = []): BoostState {
  let s = startBoosting(train, test, grid);
  for (let m = 0; m < rounds; m++) s = boostRound(s, train, test, grid, o);
  return s;
}

/** F_m(x) = F₀ + Σ η·h_k(x) over the first m rounds. */
export function predictBoosted(s: BoostState, x: number, m = s.rounds.length): number {
  let f = s.f0;
  for (let k = 0; k < m; k++) f += s.rounds[k].rate * predictValue(s.rounds[k].tree, [x]);
  return f;
}
