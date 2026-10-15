import { describe, expect, it } from 'vitest';
import { gridPoints, lucasKanade, makeFrames } from './opticalFlow';

const W = 96, H = 80;
const meanError = (kind: 'translate' | 'rotate' | 'zoom', amount: number, levels: number) => {
  const { frame1, frame2, truth } = makeFrames(kind, amount, W, H, 'texture');
  const flow = lucasKanade(frame1, frame2, gridPoints(W, H, 12, 16), 5, levels);
  return flow.reduce((s, f) => { const [u, v] = truth(f.x, f.y); return s + Math.hypot(f.u - u, f.v - v); }, 0) / flow.length;
};

describe('Lucas–Kanade', () => {
  it('measures a small translation to a fraction of a pixel', () => {
    expect(meanError('translate', 1.2, 1)).toBeLessThan(0.05);
  });

  it('handles rotation and zoom, with a small bias because the motion varies inside the window', () => {
    expect(meanError('rotate', 2, 2)).toBeLessThan(0.3);
    expect(meanError('zoom', 3, 2)).toBeLessThan(0.15);
  });

  it('fails on a large motion without a pyramid, succeeds with one', () => {
    expect(meanError('translate', 10, 1)).toBeGreaterThan(1);
    expect(meanError('translate', 10, 3)).toBeLessThan(0.1);
  });

  it('reports the aperture problem on stripes', () => {
    const tex = makeFrames('translate', 1, W, H, 'texture');
    const str = makeFrames('translate', 1, W, H, 'stripes');
    const [t] = lucasKanade(tex.frame1, tex.frame2, [{ x: 48, y: 40 }], 5, 1);
    const [s] = lucasKanade(str.frame1, str.frame2, [{ x: 48, y: 40 }], 5, 1);
    expect(s.minEigen).toBeLessThan(1e-6);
    expect(t.minEigen).toBeGreaterThan(1);
    expect(s.u).toBeCloseTo(1, 1); // the motion across the stripes is still measured
  });
});
