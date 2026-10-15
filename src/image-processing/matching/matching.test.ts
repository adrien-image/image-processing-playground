import { describe, expect, it } from 'vitest';
import { detectHarris } from './harris';
import { matchKeypoints } from './descriptors';
import { applySimilarity, estimateSimilarity, ransacSimilarity, type Point, type Similarity } from './transform';
import { seededRandom } from '../../lib/random';
import { SCENE_HEIGHT, SCENE_WIDTH, groundTruth, type SceneTransform } from './scene';

// ---- A tiny rasteriser for the demo scene (no canvas needed in Node) ----

type Shape =
  | { kind: 'rect'; x: number; y: number; w: number; h: number; v: number }
  | { kind: 'circle'; cx: number; cy: number; r: number; v: number };

// Same layout as scene.ts draws on the canvas.
const SHAPES: Shape[] = [
  { kind: 'rect', x: 40, y: 40, w: 120, h: 80, v: 221 },
  { kind: 'rect', x: 60, y: 60, w: 60, h: 60, v: 136 },
  { kind: 'circle', cx: 180, cy: 60, r: 30, v: 170 },
  { kind: 'circle', cx: 180, cy: 120, r: 20, v: 102 },
  { kind: 'rect', x: 30, y: 150, w: 140, h: 50, v: 255 },
  { kind: 'rect', x: 32, y: 152, w: 136, h: 46, v: 17 },
  { kind: 'rect', x: 50, y: 160, w: 30, h: 30, v: 153 },
];

function sceneValue(x: number, y: number): number {
  let v = 17;
  for (const s of SHAPES) {
    const inside = s.kind === 'rect'
      ? x >= s.x && x < s.x + s.w && y >= s.y && y < s.y + s.h
      : (x - s.cx) ** 2 + (y - s.cy) ** 2 <= s.r * s.r;
    if (inside) v = s.v;
  }
  return v;
}

/** Render the scene through `t` (canvas semantics: rotate/scale about the centre), 4×4 supersampled. */
function render(t: SceneTransform | null): Float64Array {
  const W = SCENE_WIDTH, H = SCENE_HEIGHT, out = new Float64Array(W * H);
  const c = t ? Math.cos(-t.rotate) : 1, s = t ? Math.sin(-t.rotate) : 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let sy = 0; sy < 4; sy++) for (let sx = 0; sx < 4; sx++) {
        let px = x + (sx + 0.5) / 4, py = y + (sy + 0.5) / 4;
        if (t) {
          const dx = px - W / 2 - t.tx, dy = py - H / 2 - t.ty;
          px = (c * dx - s * dy) / t.scale + W / 2;
          py = (s * dx + c * dy) / t.scale + H / 2;
        }
        acc += sceneValue(px, py);
      }
      out[y * W + x] = acc / 16;
    }
  }
  return out;
}

function runPipeline(t: SceneTransform) {
  const a = render(null), b = render(t);
  const kpsA = detectHarris(a, SCENE_WIDTH, SCENE_HEIGHT, 500, 0.06);
  const kpsB = detectHarris(b, SCENE_WIDTH, SCENE_HEIGHT, 500, 0.06);
  const matches = matchKeypoints(a, SCENE_WIDTH, SCENE_HEIGHT, kpsA, b, SCENE_WIDTH, SCENE_HEIGHT, kpsB);
  const src = matches.map(m => kpsA[m.srcIdx]), dst = matches.map(m => kpsB[m.dstIdx]);
  return { matches, result: ransacSimilarity(src, dst) };
}

const deg = (r: number) => (r * 180) / Math.PI;

// ---- Tests ----

describe('estimateSimilarity', () => {
  it('recovers an exact transform including scale', () => {
    const t: Similarity = { angle: 0.4, scale: 0.85, tx: 30, ty: -12 };
    const src: Point[] = [{ x: 0, y: 0 }, { x: 10, y: 3 }, { x: -4, y: 8 }];
    const est = estimateSimilarity(src, src.map(p => applySimilarity(t, p)))!;
    expect(est.angle).toBeCloseTo(0.4, 9);
    expect(est.scale).toBeCloseTo(0.85, 9);
    expect(est.tx).toBeCloseTo(30, 9);
    expect(est.ty).toBeCloseTo(-12, 9);
  });
});

describe('ransacSimilarity', () => {
  it('ignores outliers that would ruin a plain least-squares fit', () => {
    const t: Similarity = { angle: -0.3, scale: 1.2, tx: 5, ty: 9 };
    const rand = seededRandom(1);
    const src = Array.from({ length: 30 }, () => ({ x: rand() * 200, y: rand() * 200 }));
    const dst = src.map(p => applySimilarity(t, p));
    for (let i = 0; i < 8; i++) dst[i] = { x: rand() * 200, y: rand() * 200 }; // 8 wrong matches

    const ls = estimateSimilarity(src, dst)!;
    expect(Math.abs(ls.angle - t.angle)).toBeGreaterThan(0.01);

    const r = ransacSimilarity(src, dst)!;
    expect(r.inlierCount).toBe(22);
    expect(r.inliers.slice(0, 8).some(Boolean)).toBe(false);
    expect(r.transform.angle).toBeCloseTo(t.angle, 6);
    expect(r.transform.scale).toBeCloseTo(t.scale, 6);
  });
});

describe('detectHarris', () => {
  it('finds the corners of a rectangle and nothing on a flat image', () => {
    const w = 40, h = 40, img = new Float64Array(w * h);
    for (let y = 10; y < 30; y++) for (let x = 10; x < 30; x++) img[y * w + x] = 200;
    const kps = detectHarris(img, w, h, 10);
    expect(kps.length).toBeGreaterThanOrEqual(4);
    for (const [cx, cy] of [[10, 10], [29, 10], [10, 29], [29, 29]]) {
      expect(kps.some(k => Math.hypot(k.x - cx, k.y - cy) <= 2)).toBe(true);
    }
    expect(detectHarris(new Float64Array(w * h).fill(80), w, h)).toEqual([]);
  });
});

describe('full pipeline on the demo scene', () => {
  it('recovers the default ground-truth transform', () => {
    const t: SceneTransform = { rotate: 0.4, scale: 0.85, tx: 8, ty: -6 };
    const { matches, result } = runPipeline(t);
    const gt = groundTruth(t);
    expect(matches.length).toBeGreaterThanOrEqual(6);
    expect(result).not.toBeNull();
    expect(Math.abs(deg(result!.transform.angle) - deg(gt.angle))).toBeLessThan(1.5);
    expect(Math.abs(result!.transform.scale - gt.scale)).toBeLessThan(0.03);
    expect(Math.abs(result!.transform.tx - gt.tx)).toBeLessThan(3);
    expect(Math.abs(result!.transform.ty - gt.ty)).toBeLessThan(3);
  });

  it('works for a larger rotation that unrotated patches could not match', () => {
    const t: SceneTransform = { rotate: -0.9, scale: 0.9, tx: -5, ty: 4 };
    const { result } = runPipeline(t);
    const gt = groundTruth(t);
    expect(result).not.toBeNull();
    expect(Math.abs(deg(result!.transform.angle) - deg(gt.angle))).toBeLessThan(2);
    expect(Math.abs(result!.transform.scale - gt.scale)).toBeLessThan(0.04);
  });
});
