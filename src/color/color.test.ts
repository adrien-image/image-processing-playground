import { describe, expect, it } from 'vitest';
import { adjustColors, extractChannel, hsvToRgb, NO_ADJUSTMENT, rgbToHsv } from './color';

const px = (...rgb: number[]) => Uint8ClampedArray.from([...rgb, 255]);

describe('HSV conversion', () => {
  it('round-trips arbitrary colours', () => {
    for (const [r, g, b] of [[255, 51, 51], [51, 255, 51], [12, 34, 200], [0, 0, 0], [255, 255, 255], [128, 128, 64]]) {
      const back = hsvToRgb(...rgbToHsv(r, g, b));
      expect(back.map(Math.round)).toEqual([r, g, b]);
    }
  });
});

describe('adjustColors', () => {
  it('is the identity with no adjustment', () => {
    expect(Array.from(adjustColors(px(10, 120, 240), NO_ADJUSTMENT))).toEqual([10, 120, 240, 255]);
  });
  it('desaturates to gray', () => {
    const [r, g, b] = adjustColors(px(255, 51, 51), { ...NO_ADJUSTMENT, satScale: 0 });
    expect(r).toBe(g);
    expect(g).toBe(b);
  });
  it('rotates the hue by 120° (red → green)', () => {
    expect(Array.from(adjustColors(px(255, 0, 0), { ...NO_ADJUSTMENT, hueShift: 120 })).slice(0, 3)).toEqual([0, 255, 0]);
  });
  it('gamma < 1 darkens mid-tones', () => {
    expect(adjustColors(px(128, 128, 128), { ...NO_ADJUSTMENT, gamma: 0.5 })[0]).toBeLessThan(128);
  });
});

describe('extractChannel', () => {
  it('V is the max channel: all saturated demo colours are equally bright', () => {
    for (const c of [[255, 51, 51], [51, 255, 51], [51, 51, 255], [255, 255, 51]]) {
      expect(extractChannel(px(...c), 'v')[0]).toBe(255);
    }
  });
  it('S is 0 for gray', () => {
    expect(extractChannel(px(90, 90, 90), 's')[0]).toBe(0);
  });
});
