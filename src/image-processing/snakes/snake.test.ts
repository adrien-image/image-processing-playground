import { describe, expect, it } from 'vitest';
import { circle, forceField, perimeter, resample, snakeMatrix, snakeStep, type Point } from './snake';

const S = 64;
/** A bright disc of radius 14 centred at (32, 30). */
const disc = Float64Array.from({ length: S * S }, (_, i) => (Math.hypot((i % S) - 32, Math.floor(i / S) - 30) < 14 ? 220 : 30));
const meanRadius = (pts: Point[]) => pts.reduce((s, p) => s + Math.hypot(p.x - 32, p.y - 30), 0) / pts.length;

function run(points: Point[], steps: number, opts: { alpha: number; beta: number; tau: number; edgeWeight: number; balloon: number }) {
  const field = forceField(disc, S, S, 1.5);
  const inv = snakeMatrix(points.length, opts.alpha, opts.beta, opts.tau);
  let p = points;
  for (let i = 0; i < steps; i++) {
    p = snakeStep(p, inv, field, opts);
    if (i % 10 === 9) p = resample(p);
  }
  return p;
}

describe('snakeMatrix', () => {
  it('is the inverse of I + τA', () => {
    const n = 12, alpha = 0.3, beta = 0.2, tau = 2;
    const inv = snakeMatrix(n, alpha, beta, tau);
    const stencil = [beta, -alpha - 4 * beta, 2 * alpha + 6 * beta, -alpha - 4 * beta, beta];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        let s = 0;
        for (let k = -2; k <= 2; k++) s += tau * stencil[k + 2] * inv[((r + k + n) % n) * n + c];
        s += inv[r * n + c];
        expect(s).toBeCloseTo(r === c ? 1 : 0, 10);
      }
    }
  });
});

describe('snake', () => {
  it('shrinks under elasticity alone and stays round', () => {
    const start = circle(32, 30, 25, 40);
    const end = run(start, 50, { alpha: 0.2, beta: 0.1, tau: 1, edgeWeight: 0, balloon: 0 });
    expect(perimeter(end)).toBeLessThan(perimeter(start) * 0.95);
    const r = meanRadius(end);
    for (const p of end) expect(Math.abs(Math.hypot(p.x - 32, p.y - 30) - r)).toBeLessThan(0.5);
  });

  it('deflates onto a disc from outside', () => {
    const end = run(circle(32, 30, 26, 48), 300, { alpha: 0.05, beta: 0.05, tau: 1, edgeWeight: 3, balloon: -0.4 });
    expect(Math.abs(meanRadius(end) - 14)).toBeLessThan(1.5);
  });

  it('inflates onto the disc boundary from inside', () => {
    const end = run(circle(30, 31, 4, 32), 300, { alpha: 0.05, beta: 0.05, tau: 1, edgeWeight: 3, balloon: 0.4 });
    expect(Math.abs(meanRadius(end) - 14)).toBeLessThan(1.5);
  });
});

describe('resample', () => {
  it('spaces points evenly along the curve', () => {
    const uneven = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }];
    const even = resample(uneven, 40);
    const gaps = even.map((p, i) => Math.hypot(even[(i + 1) % 40].x - p.x, even[(i + 1) % 40].y - p.y));
    for (const g of gaps) expect(g).toBeCloseTo(1, 6);
  });
});
