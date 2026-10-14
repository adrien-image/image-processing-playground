import { describe, expect, it } from 'vitest';
import { extractPatch, nccMatch } from './ncc';
import { seededRandom } from '../lib/random';

const W = 40, H = 30;
const rand = seededRandom(11);
const NOISE = Float64Array.from({ length: 4 * W * H }, () => Math.floor(rand() * 256));
const texture = (i: number) => NOISE[i];

describe('nccMatch', () => {
  it('finds a copied patch exactly, with score 1, even after a brightness/contrast change', () => {
    const scene = Float64Array.from({ length: W * H }, (_, i) => texture(i));
    const patch = extractPatch(scene, W, 13, 9, 8, 6);
    const changed = patch.map(v => v * 0.5 + 40);
    const m = nccMatch(scene, W, H, changed, 8, 6)!;
    expect([m.x, m.y]).toEqual([13, 9]);
    expect(m.score).toBeCloseTo(1, 9);
  });

  it('matches the brute-force definition at every placement', () => {
    const scene = Float64Array.from({ length: W * H }, (_, i) => texture(i * 3));
    const t = Float64Array.from({ length: 20 }, (_, i) => texture(i + 7));
    const m = nccMatch(scene, W, H, t, 5, 4)!;
    const tm = t.reduce((a, b) => a + b, 0) / 20;
    for (const [x, y] of [[0, 0], [17, 11], [W - 5, H - 4]]) {
      const win = extractPatch(scene, W, x, y, 5, 4);
      const wm = win.reduce((a, b) => a + b, 0) / 20;
      let num = 0, a = 0, b = 0;
      for (let i = 0; i < 20; i++) { num += (win[i] - wm) * (t[i] - tm); a += (win[i] - wm) ** 2; b += (t[i] - tm) ** 2; }
      expect(m.scores[y * m.scoresWidth + x]).toBeCloseTo(num / Math.sqrt(a * b), 9);
    }
  });

  it('refuses a flat template', () => {
    expect(nccMatch(new Float64Array(W * H), W, H, new Float64Array(12).fill(26), 4, 3)).toBeNull();
  });
});
