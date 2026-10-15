import { describe, expect, it } from 'vitest';
import { hotColor, houghCircles, houghLines } from './hough';

const W = 160, H = 120;
const blank = () => new Float64Array(W * H);
function drawSegment(img: Float64Array, x1: number, y1: number, x2: number, y2: number, skip: (t: number) => boolean = () => false) {
  const n = Math.ceil(Math.hypot(x2 - x1, y2 - y1));
  for (let i = 0; i <= n; i++) {
    if (skip(i)) continue;
    const x = Math.round(x1 + ((x2 - x1) * i) / n), y = Math.round(y1 + ((y2 - y1) * i) / n);
    img[y * W + x] = 255;
  }
}

describe('hotColor', () => {
  it('maps 0 → black and 1 → white, always within 0–255', () => {
    expect(hotColor(0)).toEqual([0, 0, 0]);
    expect(hotColor(1)).toEqual([255, 255, 255]);
    for (let t = 0; t <= 1; t += 0.05) for (const c of hotColor(t)) expect(c).toBeLessThanOrEqual(255);
    // Mid values are red/orange, not saturated yellow (regression: green was multiplied by 255 twice).
    expect(hotColor(0.4)[1]).toBeLessThan(255);
  });
});

describe('houghLines', () => {
  it('finds a horizontal segment and its real endpoints', () => {
    const img = blank();
    drawSegment(img, 20, 30, 150, 30);
    const { lines } = houghLines(img, W, H, { threshold: 50, minLength: 20 });
    expect(lines).toHaveLength(1);
    const [l] = lines;
    expect((l.theta * 180) / Math.PI).toBeCloseTo(90, 0);
    expect(l.rho).toBe(30);
    expect(Math.min(l.x1, l.x2)).toBeCloseTo(20, 0);
    expect(Math.max(l.x1, l.x2)).toBeCloseTo(150, 0);
  });

  it('minLength filters by the detected segment, not the line across the image', () => {
    const img = blank();
    drawSegment(img, 10, 10, 30, 110); // long chord through the image…
    drawSegment(img, 60, 60, 120, 60, i => i > 25 && i < 40); // …and a segment with a 14-px gap
    const short = houghLines(img, W, H, { threshold: 20, minLength: 70, maxGap: 4 });
    expect(short.lines.every(l => l.length >= 70)).toBe(true);
    expect(short.lines.some(l => Math.abs(l.y1 - 60) < 1)).toBe(false);
    const bridged = houghLines(img, W, H, { threshold: 20, minLength: 50, maxGap: 16 });
    expect(bridged.lines.some(l => Math.abs(l.y1 - 60) < 1 && l.length >= 59)).toBe(true);
  });

  it('merges the two edges of a thick stroke into one line', () => {
    const img = blank();
    drawSegment(img, 20, 40, 140, 70);
    drawSegment(img, 20, 43, 140, 73);
    const { lines } = houghLines(img, W, H, { threshold: 40, minLength: 50 });
    expect(lines).toHaveLength(1);
  });

  it('treats θ near 0° and 180° as neighbours (one peak for a near-vertical line)', () => {
    const img = blank();
    drawSegment(img, 80, 5, 81, 115);
    const { lines } = houghLines(img, W, H, { threshold: 40, minLength: 20 });
    expect(lines).toHaveLength(1);
  });
});

describe('houghCircles', () => {
  it('finds a circle from edge pixels with radial gradients, once', () => {
    const img = blank(), gx = blank(), gy = blank();
    const cx = 90, cy = 60, r = 25;
    for (let a = 0; a < 360; a += 0.5) {
      const x = Math.round(cx + r * Math.cos((a * Math.PI) / 180));
      const y = Math.round(cy + r * Math.sin((a * Math.PI) / 180));
      const i = y * W + x;
      img[i] = 255;
      gx[i] = x - cx;
      gy[i] = y - cy;
    }
    const { circles } = houghCircles(img, gx, gy, W, H, { minR: 10, maxR: 40, threshold: 0.3 });
    expect(circles).toHaveLength(1);
    expect(circles[0]).toMatchObject({ x: cx, y: cy, r });
    expect(circles[0].score).toBeGreaterThan(0.5);
  });
});
