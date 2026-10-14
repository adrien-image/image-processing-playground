import { describe, expect, it } from 'vitest';
import { canny, detectEdges, hysteresis, type EdgeMethod } from './edgeDetectors';

const W = 128, H = 128;
function disk(r = 30, cx = 64, cy = 64): Float64Array {
  const img = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) img[y * W + x] = Math.hypot(x - cx, y - cy) <= r ? 221 : 51;
  return img;
}

/** Edge pixels and 8-connected fragments within an angular sector of the disk. */
function sectorStats(edges: Float64Array, fromDeg: number, toDeg: number) {
  const inSector = (x: number, y: number) => {
    const a = (Math.atan2(y - 64, x - 64) * 180) / Math.PI;
    return a > fromDeg && a < toDeg;
  };
  const seen = new Uint8Array(W * H);
  let fragments = 0, pixels = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (edges[i] !== 255 || !inSector(x, y) || seen[i]) continue;
    fragments++;
    const stack = [i];
    seen[i] = 1;
    while (stack.length) {
      const j = stack.pop()!;
      pixels++;
      const jx = j % W, jy = (j - jx) / W;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const k = (jy + dy) * W + jx + dx;
        if (edges[k] === 255 && inSector(jx + dx, jy + dy) && !seen[k]) { seen[k] = 1; stack.push(k); }
      }
    }
  }
  return { fragments, pixels };
}

describe('canny', () => {
  it('traces diagonal arcs as one thin, unbroken edge (regression: diagonals were fragmented)', () => {
    const edges = canny(disk(), W, H, 20, 50);
    for (const [a, b] of [[25, 65], [115, 155], [-65, -25], [-155, -115]]) {
      const s = sectorStats(edges, a, b);
      expect(s.fragments).toBe(1);
      // A thin 8-connected arc of radius 30 over 40° has ~21–30 pixels.
      expect(s.pixels).toBeGreaterThan(15);
      expect(s.pixels).toBeLessThan(35);
    }
  });

  it('keeps axis-aligned parts of the circle one pixel thick', () => {
    const edges = canny(disk(), W, H, 20, 50);
    let onRow = 0;
    for (let x = 64; x < W; x++) if (edges[64 * W + x] === 255) onRow++;
    expect(onRow).toBe(1);
  });
});

describe('hysteresis', () => {
  it('follows weak chains in every direction from a strong pixel', () => {
    const w = 12, h = 12, nms = new Float64Array(w * h);
    nms[10 * w + 10] = 100; // strong
    for (let k = 1; k <= 9; k++) nms[(10 - k) * w + (10 - k)] = 30; // weak chain going up-left
    nms[11] = 30; // isolated weak pixel in the top row
    const out = hysteresis(nms, w, h, 20, 80);
    for (let k = 0; k <= 9; k++) expect(out[(10 - k) * w + (10 - k)]).toBe(255);
    expect(out[11]).toBe(0);
  });
});

describe('detectEdges', () => {
  it('finds no edges on a flat image — no border frame', () => {
    const flat = new Float64Array(W * H).fill(51);
    for (const m of ['sobel-gx', 'sobel-gy', 'sobel-mag', 'laplacian', 'prewitt', 'canny'] as EdgeMethod[]) {
      expect(Math.max(...detectEdges(flat, W, H, m))).toBe(0);
    }
  });
});
