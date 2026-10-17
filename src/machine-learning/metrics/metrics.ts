/**
 * Evaluation metrics for a binary classifier, from scratch.
 *
 * A classifier gives every example a score (here a probability); a threshold t turns it
 * into a decision, positive if score ≥ t. Against the true labels that gives four counts,
 * the confusion matrix:
 *   TP  positives called positive      FN  positives called negative (misses)
 *   FP  negatives called positive      TN  negatives called negative
 * and every single-threshold metric is a ratio of them:
 *   accuracy     (TP + TN) / N        how often is it right?
 *   precision    TP / (TP + FP)       when it says positive, how often is it?
 *   recall       TP / (TP + FN)       of the real positives, how many does it find? (TPR)
 *   specificity  TN / (TN + FP)       of the real negatives, how many does it clear? (1 − FPR)
 *   F1           2PR / (P + R)        the harmonic mean of precision and recall
 *
 * Curves sweep the threshold over every distinct score, from the highest down:
 *   ROC  (FPR, TPR) from (0, 0) to (1, 1); its area, AUC, is the probability that a random
 *        positive scores above a random negative (the Mann–Whitney statistic).
 *   PR   (recall, precision); summarised by the average precision
 *        AP = Σₙ (Rₙ − Rₙ₋₁) Pₙ,  as in scikit-learn's average_precision_score.
 * A ratio with a zero denominator (e.g. precision when nothing is called positive) is NaN.
 */
import { gaussianRandom, seededRandom } from '../../lib/random';
import { fitLogistic, predictProba } from '../linear/linear';

export interface Confusion {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

/** The confusion matrix at threshold t: predict positive iff score ≥ t. Labels are 0 / 1. */
export function confusionAt(scores: ArrayLike<number>, labels: ArrayLike<number>, t: number): Confusion {
  const c: Confusion = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (let i = 0; i < labels.length; i++) {
    const predicted = scores[i] >= t;
    if (labels[i]) { if (predicted) c.tp++; else c.fn++; }
    else if (predicted) c.fp++;
    else c.tn++;
  }
  return c;
}

export interface Rates {
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  /** False-positive rate, 1 − specificity: the ROC's x axis. */
  fpr: number;
}

const ratio = (a: number, b: number) => (b > 0 ? a / b : NaN);

export function ratesOf({ tp, fp, fn, tn }: Confusion): Rates {
  const precision = ratio(tp, tp + fp), recall = ratio(tp, tp + fn);
  return {
    accuracy: ratio(tp + tn, tp + fp + fn + tn),
    precision,
    recall,
    specificity: ratio(tn, tn + fp),
    // 2PR / (P + R) = 2TP / (2TP + FP + FN): defined even when precision is not.
    f1: ratio(2 * tp, 2 * tp + fp + fn),
    fpr: ratio(fp, fp + tn),
  };
}

/** The confusion counts at every distinct score, highest first: thresholds[k] admits every score ≥ it. */
export interface Sweep {
  thresholds: number[];
  /** Cumulative true and false positives at each threshold. */
  tp: number[];
  fp: number[];
  positives: number;
  negatives: number;
}

/**
 * Sort the scores once, highest first, and walk down: each step lowers the threshold to
 * the next distinct score, admitting every example with that score (ties move together).
 */
export function sweep(scores: ArrayLike<number>, labels: ArrayLike<number>): Sweep {
  const order = Array.from({ length: labels.length }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const out: Sweep = { thresholds: [], tp: [], fp: [], positives: 0, negatives: 0 };
  let tp = 0, fp = 0;
  for (let k = 0; k < order.length; k++) {
    const i = order[k];
    if (labels[i]) tp++; else fp++;
    if (k === order.length - 1 || scores[order[k + 1]] !== scores[i]) {
      out.thresholds.push(scores[i]);
      out.tp.push(tp);
      out.fp.push(fp);
    }
  }
  out.positives = tp;
  out.negatives = fp;
  return out;
}

export interface CurvePoint {
  x: number;
  y: number;
  /** The threshold that gives this point (+∞ for the ROC's starting corner). */
  threshold: number;
}

/** The ROC curve: (FPR, TPR) at every threshold, starting at (0, 0) with threshold +∞. */
export function rocCurve(s: Sweep): CurvePoint[] {
  const points: CurvePoint[] = [{ x: 0, y: 0, threshold: Infinity }];
  s.thresholds.forEach((t, k) => points.push({ x: ratio(s.fp[k], s.negatives), y: ratio(s.tp[k], s.positives), threshold: t }));
  return points;
}

/** Area under a curve by the trapezoid rule (points in increasing x). */
export function trapezoid(points: readonly { x: number; y: number }[]): number {
  let area = 0;
  for (let k = 1; k < points.length; k++) area += ((points[k].x - points[k - 1].x) * (points[k].y + points[k - 1].y)) / 2;
  return area;
}

/** ROC AUC: the trapezoid area under the ROC curve. A tie between a positive and a negative counts ½. */
export const rocAuc = (s: Sweep): number => trapezoid(rocCurve(s));

/**
 * The precision–recall curve: (recall, precision) at every threshold, highest first,
 * after the conventional starting point (recall 0, precision 1).
 */
export function prCurve(s: Sweep): CurvePoint[] {
  const points: CurvePoint[] = [{ x: 0, y: 1, threshold: Infinity }];
  s.thresholds.forEach((t, k) => points.push({ x: ratio(s.tp[k], s.positives), y: s.tp[k] / (s.tp[k] + s.fp[k]), threshold: t }));
  return points;
}

/**
 * Average precision, AP = Σₙ (Rₙ − Rₙ₋₁) Pₙ over the thresholds from the highest down
 * (R₀ = 0): the area under the PR curve drawn as a staircase, without the optimistic
 * interpolation a trapezoid would add.
 */
export function averagePrecision(s: Sweep): number {
  let ap = 0, previous = 0;
  for (let k = 0; k < s.thresholds.length; k++) {
    const recall = s.tp[k] / s.positives, precision = s.tp[k] / (s.tp[k] + s.fp[k]);
    ap += (recall - previous) * precision;
    previous = recall;
  }
  return s.positives ? ap : NaN;
}

/** P(score of a random positive > score of a random negative) + ½ P(tie), by checking every pair. */
export function mannWhitney(scores: ArrayLike<number>, labels: ArrayLike<number>): number {
  const pos: number[] = [], neg: number[] = [];
  for (let i = 0; i < labels.length; i++) (labels[i] ? pos : neg).push(scores[i]);
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return ratio(wins, pos.length * neg.length);
}

/**
 * The threshold (among the sweep's, plus +∞ = "call nothing positive") that minimises
 * the total cost  costFn·FN + costFp·FP.
 */
export function cheapestThreshold(s: Sweep, costFn: number, costFp = 1): { threshold: number; cost: number } {
  let best = { threshold: Infinity, cost: costFn * s.positives };
  s.thresholds.forEach((t, k) => {
    const cost = costFn * (s.positives - s.tp[k]) + costFp * s.fp[k];
    if (cost < best.cost) best = { threshold: t, cost };
  });
  return best;
}

// ---- The lesson's problem: imbalanced, overlapping classes, scored by logistic regression ----

/**
 * n points in 2-D: round(n · positiveRate) positives centred at +separation/2 along the
 * diagonal and the rest negatives at −separation/2, both with unit variance. `separation`
 * is the distance between the centres in standard deviations (d′ in signal detection).
 */
export function makeProblem(n: number, positiveRate: number, separation: number, random: () => number): { xy: number[]; labels: number[] } {
  const positives = Math.round(n * positiveRate), u = separation / 2 / Math.SQRT2;
  const xy: number[] = [], labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i < positives ? 1 : 0, c = label ? u : -u;
    xy.push(c + gaussianRandom(random), c + gaussianRandom(random));
    labels.push(label);
  }
  return { xy, labels };
}

/** Train logistic regression on one sample, then score a fresh test sample from the same distribution. */
export function scoredTestSet(positiveRate: number, separation: number, { n = 2000, seed = 1 } = {}): { scores: Float64Array; labels: number[] } {
  const train = makeProblem(n, positiveRate, separation, seededRandom(seed));
  const test = makeProblem(n, positiveRate, separation, seededRandom(seed + 1000));
  const model = fitLogistic(train.xy, train.labels, 2, { lr: 1, steps: 400 });
  return { scores: predictProba(model, test.xy, 2), labels: test.labels };
}
