import { describe, expect, it } from 'vitest';
import { equalize } from './equalize';

describe('equalize', () => {
  it('stretches a narrow range to the full 0–255 range', () => {
    const px = Float64Array.from({ length: 100 }, (_, i) => 100 + (i % 20));
    const { output } = equalize(px);
    expect(Math.min(...output)).toBe(0);
    expect(Math.max(...output)).toBe(255);
  });

  it('produces a monotonic mapping', () => {
    const px = Float64Array.from({ length: 500 }, (_, i) => (i * 7) % 256);
    const { mapping } = equalize(px);
    for (let v = 1; v < 256; v++) expect(mapping[v]).toBeGreaterThanOrEqual(mapping[v - 1]);
  });

  it('leaves a single-level image unchanged', () => {
    const { output } = equalize(new Float64Array(10).fill(77));
    expect(Array.from(output).every(v => v === 77)).toBe(true);
  });
});
