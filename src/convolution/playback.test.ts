import { describe, expect, it } from 'vitest';
import { revealPartial, snakeOrder, stepMath } from './playback';
import { applyProcessor } from '../processors';
import { SharpenProcessor } from '../processors/sharpen';
import { makeOutputMapper } from '../processors/utils';

describe('snakeOrder', () => {
  it('covers every pixel exactly once, including the border', () => {
    const order = snakeOrder(4, 3);
    expect(order).toHaveLength(12);
    expect(new Set(order.map(p => `${p.col},${p.row}`)).size).toBe(12);
    expect(order.slice(0, 5)).toEqual([
      { col: 0, row: 0 }, { col: 1, row: 0 }, { col: 2, row: 0 }, { col: 3, row: 0 }, { col: 3, row: 1 },
    ]);
  });
});

describe('stepMath', () => {
  it('matches the full-image result for every pixel (border included)', () => {
    const w = 6, h = 5;
    const gray = Float64Array.from({ length: w * h }, (_, i) => (i * 37) % 256);
    const { raw, output, kernel } = applyProcessor(SharpenProcessor, gray, w, h, { amount: 1.5 });
    const map = makeOutputMapper(raw, 'clamp');
    for (const pos of snakeOrder(w, h)) {
      const m = stepMath(gray, w, h, kernel, pos, map);
      const idx = pos.row * w + pos.col;
      expect(m.sum).toBeCloseTo(raw[idx], 9);
      expect(m.output).toBe(output[idx]);
    }
  });
});

describe('revealPartial', () => {
  it('shows only the first n pixels', () => {
    const out = Float64Array.from([1, 2, 3, 4]);
    expect(Array.from(revealPartial(out, snakeOrder(2, 2), 3, 2))).toEqual([1, 2, 0, 4]);
  });
});
