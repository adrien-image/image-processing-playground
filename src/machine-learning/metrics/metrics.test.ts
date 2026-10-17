import { describe, expect, it } from 'vitest';
import { seededRandom } from '../../lib/random';
import {
  averagePrecision, cheapestThreshold, confusionAt, makeProblem, mannWhitney, prCurve, ratesOf, rocAuc, rocCurve,
  scoredTestSet, sweep, trapezoid,
} from './metrics';

// A small case worked by hand: positives score 0.9, 0.8, 0.6; negatives 0.7, 0.55, 0.4.
const scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4];
const labels = [1, 1, 0, 1, 0, 0];

describe('confusion matrix and rates', () => {
  it('counts and ratios at threshold 0.6 (score ≥ t is positive)', () => {
    const c = confusionAt(scores, labels, 0.6);
    expect(c).toEqual({ tp: 3, fp: 1, fn: 0, tn: 2 });
    const r = ratesOf(c);
    expect(r.accuracy).toBeCloseTo(5 / 6, 12);
    expect(r.precision).toBeCloseTo(3 / 4, 12);
    expect(r.recall).toBe(1);
    expect(r.specificity).toBeCloseTo(2 / 3, 12);
    expect(r.fpr).toBeCloseTo(1 / 3, 12);
    expect(r.f1).toBeCloseTo((2 * 0.75 * 1) / 1.75, 12);
  });

  it('precision is undefined (NaN) when nothing is called positive, and F1 is then 0', () => {
    const r = ratesOf(confusionAt(scores, labels, 2));
    expect(r.precision).toBeNaN();
    expect(r.recall).toBe(0);
    expect(r.f1).toBe(0);
    expect(r.accuracy).toBe(0.5);
  });
});

describe('ROC and PR curves', () => {
  const s = sweep(scores, labels);

  it('the ROC curve of the hand case, and its area 8/9', () => {
    expect(rocCurve(s).map(p => [p.x, p.y])).toEqual([[0, 0], [0, 1 / 3], [0, 2 / 3], [1 / 3, 2 / 3], [1 / 3, 1], [2 / 3, 1], [1, 1]]);
    expect(rocAuc(s)).toBeCloseTo(8 / 9, 12);
    expect(mannWhitney(scores, labels)).toBeCloseTo(8 / 9, 12);
  });

  it('average precision of the hand case: 1/3·1 + 1/3·1 + 0·2/3 + 1/3·3/4 = 11/12 (scikit-learn gives 0.9167)', () => {
    expect(averagePrecision(s)).toBeCloseTo(11 / 12, 12);
    expect(prCurve(s).map(p => [p.x, p.y])).toEqual([[0, 1], [1 / 3, 1], [2 / 3, 1], [2 / 3, 2 / 3], [1, 3 / 4], [1, 3 / 5], [1, 1 / 2]]);
  });

  it('ties move together: one shared score makes a diagonal ROC step worth ½', () => {
    const t = sweep([0.5, 0.5], [1, 0]);
    expect(t.thresholds).toEqual([0.5]);
    expect(rocAuc(t)).toBe(0.5);
    expect(mannWhitney([0.5, 0.5], [1, 0])).toBe(0.5);
    expect(averagePrecision(t)).toBe(0.5);
  });

  it('AUC equals the Mann–Whitney probability P(score₊ > score₋) + ½ P(tie), with many ties', () => {
    for (const seed of [1, 2, 3]) {
      const random = seededRandom(seed);
      const y = Array.from({ length: 300 }, () => (random() < 0.3 ? 1 : 0));
      const sc = y.map(v => Math.round((random() + 0.3 * v) * 10) / 10); // rounded: lots of ties
      expect(rocAuc(sweep(sc, y))).toBeCloseTo(mannWhitney(sc, y), 12);
    }
  });

  it('a perfect classifier scores AUC = AP = 1; a random one AUC ≈ ½ and AP ≈ the positive rate', () => {
    const random = seededRandom(4);
    const y: number[] = Array.from({ length: 20000 }, () => (random() < 0.1 ? 1 : 0));
    const perfect = sweep(y.map(v => v + random() * 0.5), y);
    expect(rocAuc(perfect)).toBe(1);
    expect(averagePrecision(perfect)).toBe(1);
    const coin = sweep(y.map(() => random()), y);
    const rate = y.reduce((a, b) => a + b) / y.length;
    expect(rocAuc(coin)).toBeCloseTo(0.5, 1);
    expect(Math.abs(averagePrecision(coin) - rate)).toBeLessThan(0.01);
  });

  it('trapezoid integrates a straight line exactly', () => {
    expect(trapezoid([{ x: 0, y: 0 }, { x: 0.3, y: 0.3 }, { x: 1, y: 1 }])).toBeCloseTo(0.5, 12);
  });
});

describe('the lesson problem', () => {
  it('makes exactly the requested share of positives', () => {
    const { labels: y } = makeProblem(2000, 0.02, 2, seededRandom(1));
    expect(y.filter(v => v === 1)).toHaveLength(40);
  });

  it('under imbalance accuracy flatters and AP falls, while ROC AUC barely moves', () => {
    const balanced = scoredTestSet(0.5, 2), rare = scoredTestSet(0.02, 2);
    const sb = sweep(balanced.scores, balanced.labels), sr = sweep(rare.scores, rare.labels);
    expect(rocAuc(sb)).toBeCloseTo(0.921, 3);
    expect(rocAuc(sr)).toBeCloseTo(0.920, 3);
    expect(averagePrecision(sb)).toBeCloseTo(0.922, 3);
    expect(averagePrecision(sr)).toBeCloseTo(0.362, 3);
    // At 2% positives the model is exactly as accurate as always answering "negative".
    const r = ratesOf(confusionAt(rare.scores, rare.labels, 0.5));
    expect(r.accuracy).toBeCloseTo(0.98, 3);
    expect(r.recall).toBeCloseTo(0.15, 3);
  });

  it('the default view (10% positives, d′ = 2) at threshold 0.5', () => {
    const { scores: sc, labels: y } = scoredTestSet(0.1, 2);
    const r = ratesOf(confusionAt(sc, y, 0.5));
    expect(r.accuracy).toBeCloseTo(0.923, 3);
    expect(r.recall).toBeCloseTo(0.43, 3);
    expect(rocAuc(sweep(sc, y))).toBeCloseTo(0.925, 3);
    expect(averagePrecision(sweep(sc, y))).toBeCloseTo(0.639, 3);
  });

  it('for a miss costing 10 false alarms, the cheapest threshold is near 1/(1 + 10) for these calibrated scores', () => {
    const { scores: sc, labels: y } = scoredTestSet(0.1, 2);
    const best = cheapestThreshold(sweep(sc, y), 10);
    expect(Math.abs(best.threshold - 1 / 11)).toBeLessThan(0.03);
  });
});
