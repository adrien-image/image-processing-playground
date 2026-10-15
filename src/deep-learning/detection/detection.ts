/**
 * The bookkeeping around an object detector — everything except the network itself:
 *
 *   iou               — how much two boxes overlap (intersection over union)
 *   nonMaxSuppression — keep the best box per object, drop its near-duplicates
 *   evaluate          — match detections to ground truth: precision, recall, average precision
 *   anchors           — the fixed reference boxes that detectors like SSD, RetinaNet and
 *                       Faster R-CNN predict offsets from, and how they are labelled for training
 */
import { gaussianRandom } from '../../lib/random';

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Detection extends Box {
  score: number;
}

export const area = (b: Box) => Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0);

/** Intersection over union: 0 for disjoint boxes, 1 for identical ones. */
export function iou(a: Box, b: Box): number {
  const inter = area({ x0: Math.max(a.x0, b.x0), y0: Math.max(a.y0, b.y0), x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1) });
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
}

// ---------------------------------------------------------------- NMS

export interface NmsStep {
  /** The box kept at this step (the highest-scoring one still in play). */
  kept: number;
  /** Boxes it suppressed because they overlap it by more than the threshold. */
  suppressed: number[];
}

/**
 * Greedy non-maximum suppression. Boxes scoring below `scoreThreshold` are dropped
 * first; then, repeatedly, the best remaining box is kept and every remaining box
 * overlapping it by IoU > `iouThreshold` is removed. Returns the steps in order.
 */
export function nonMaxSuppression(dets: Detection[], iouThreshold: number, scoreThreshold = 0): NmsStep[] {
  let remaining = dets.map((_, i) => i).filter(i => dets[i].score >= scoreThreshold).sort((a, b) => dets[b].score - dets[a].score);
  const steps: NmsStep[] = [];
  while (remaining.length > 0) {
    const [best, ...rest] = remaining;
    const suppressed = rest.filter(i => iou(dets[best], dets[i]) > iouThreshold);
    steps.push({ kept: best, suppressed });
    remaining = rest.filter(i => !suppressed.includes(i));
  }
  return steps;
}

// ---------------------------------------------------------------- evaluation

export interface Evaluation {
  /** Per detection, in descending score order: was it a true positive? */
  ranked: { index: number; truePositive: boolean }[];
  /** Precision and recall after each ranked detection. */
  curve: { precision: number; recall: number }[];
  /** Area under the interpolated precision–recall curve (PASCAL VOC, all points). */
  averagePrecision: number;
  truePositives: number;
  falsePositives: number;
  missed: number;
}

/**
 * Match detections to ground-truth boxes the standard way: in descending score
 * order, a detection is a true positive if it overlaps a not-yet-matched ground-truth
 * box with IoU ≥ `iouMatch`; otherwise (no overlap, or a duplicate) it is a false positive.
 */
export function evaluate(dets: Detection[], indices: number[], truth: Box[], iouMatch = 0.5): Evaluation {
  const order = [...indices].sort((a, b) => dets[b].score - dets[a].score);
  const matched = new Set<number>();
  const ranked = order.map(index => {
    let best = -1;
    let bestIou = iouMatch;
    truth.forEach((t, g) => {
      const o = iou(dets[index], t);
      if (!matched.has(g) && o >= bestIou) { best = g; bestIou = o; }
    });
    if (best >= 0) matched.add(best);
    return { index, truePositive: best >= 0 };
  });

  let tp = 0;
  const curve = ranked.map((r, k) => {
    if (r.truePositive) tp++;
    return { precision: tp / (k + 1), recall: truth.length ? tp / truth.length : 0 };
  });

  // Interpolate: precision at recall r is the best precision at any recall ≥ r.
  let ap = 0;
  let prevRecall = 0;
  for (let k = 0; k < curve.length; k++) {
    let p = 0;
    for (let m = k; m < curve.length; m++) p = Math.max(p, curve[m].precision);
    ap += (curve[k].recall - prevRecall) * p;
    prevRecall = curve[k].recall;
  }
  return { ranked, curve, averagePrecision: ap, truePositives: tp, falsePositives: ranked.length - tp, missed: truth.length - tp };
}

// ---------------------------------------------------------------- anchors

export interface Anchor extends Box {
  cellX: number;
  cellY: number;
  shape: number;
}

/**
 * One anchor per (grid cell, scale, aspect ratio), centred on the cell. Ratio = width / height.
 */
export function anchors(imageSize: number, grid: number, scales: number[], ratios: number[]): Anchor[] {
  const cell = imageSize / grid;
  const out: Anchor[] = [];
  for (let cy = 0; cy < grid; cy++) {
    for (let cx = 0; cx < grid; cx++) {
      const x = (cx + 0.5) * cell, y = (cy + 0.5) * cell;
      let shape = 0;
      for (const s of scales) {
        for (const r of ratios) {
          const w = s * Math.sqrt(r), h = s / Math.sqrt(r);
          out.push({ x0: x - w / 2, y0: y - h / 2, x1: x + w / 2, y1: y + h / 2, cellX: cx, cellY: cy, shape: shape++ });
        }
      }
    }
  }
  return out;
}

export type AnchorLabel = { kind: 'positive'; truth: number; iou: number } | { kind: 'negative' } | { kind: 'ignored'; iou: number };

/**
 * Label anchors for training: positive (learn to predict this object) when IoU ≥ `positive`,
 * negative (learn "background") when IoU < `negative`, ignored in between. Every object
 * also claims its single best anchor, so none goes unassigned.
 */
export function labelAnchors(list: Anchor[], truth: Box[], positive = 0.5, negative = 0.4): AnchorLabel[] {
  const labels: AnchorLabel[] = list.map(a => {
    let best = -1, bestIou = 0;
    truth.forEach((t, g) => { const o = iou(a, t); if (o > bestIou) { best = g; bestIou = o; } });
    if (best >= 0 && bestIou >= positive) return { kind: 'positive', truth: best, iou: bestIou };
    return bestIou < negative ? { kind: 'negative' } : { kind: 'ignored', iou: bestIou };
  });
  truth.forEach((t, g) => {
    let best = -1, bestIou = 0;
    list.forEach((a, i) => { const o = iou(a, t); if (o > bestIou) { best = i; bestIou = o; } });
    if (best >= 0) labels[best] = { kind: 'positive', truth: g, iou: bestIou };
  });
  return labels;
}

// ---------------------------------------------------------------- a simulated detector

export type ShapeKind = 'square' | 'disc' | 'triangle';

export interface SceneObject extends Box {
  kind: ShapeKind;
}

/** A few non-overlapping shapes of varied size and aspect ratio in a `size`×`size` image. */
export function makeScene(size: number, count: number, random: () => number): SceneObject[] {
  const kinds: ShapeKind[] = ['square', 'disc', 'triangle'];
  const objects: SceneObject[] = [];
  for (let attempt = 0; objects.length < count && attempt < 500; attempt++) {
    const w = size * (0.12 + random() * 0.18);
    const h = w * (0.6 + random() * 0.8);
    const x0 = random() * (size - w), y0 = random() * (size - h);
    const box = { x0, y0, x1: x0 + w, y1: y0 + h };
    if (objects.some(o => iou(o, { x0: box.x0 - 6, y0: box.y0 - 6, x1: box.x1 + 6, y1: box.y1 + 6 }) > 0)) continue;
    objects.push({ ...box, kind: kinds[objects.length % kinds.length] });
  }
  return objects;
}

/**
 * What a detector's raw output looks like before NMS: several overlapping, jittered
 * boxes around every object (scores lower the further off they are) plus a few
 * spurious low-confidence boxes.
 */
export function simulateDetections(truth: Box[], size: number, random: () => number, perObject = 6, spurious = 6): Detection[] {
  const dets: Detection[] = [];
  for (const t of truth) {
    const w = t.x1 - t.x0, h = t.y1 - t.y0;
    for (let k = 0; k < perObject; k++) {
      const j = k === 0 ? 0.04 : 0.1 + 0.08 * random();
      const box = {
        x0: t.x0 + gaussianRandom(random) * j * w, y0: t.y0 + gaussianRandom(random) * j * h,
        x1: t.x1 + gaussianRandom(random) * j * w, y1: t.y1 + gaussianRandom(random) * j * h,
      };
      const score = Math.min(0.99, Math.max(0.05, 0.25 + 0.75 * iou(box, t) + 0.08 * gaussianRandom(random)));
      dets.push({ ...box, score });
    }
  }
  for (let k = 0; k < spurious; k++) {
    const w = size * (0.08 + random() * 0.2), h = size * (0.08 + random() * 0.2);
    const x0 = random() * (size - w), y0 = random() * (size - h);
    dets.push({ x0, y0, x1: x0 + w, y1: y0 + h, score: 0.05 + random() * 0.6 });
  }
  return dets;
}
