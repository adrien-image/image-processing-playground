import { describe, expect, it } from 'vitest';
import { fitWithin, quietImageData } from './canvas';

describe('quietImageData', () => {
  it('hides an own enumerable data property from for…in but keeps it readable', () => {
    const img = { data: new Uint8ClampedArray([1, 2, 3, 4]), width: 1, height: 1 };
    quietImageData(img);
    const keys: string[] = [];
    for (const k in img) keys.push(k);
    expect(keys).toEqual(['width', 'height']);
    expect(Array.from(img.data)).toEqual([1, 2, 3, 4]);
  });

  it('leaves objects without an own data property alone', () => {
    const proto = { get data() { return 1; } };
    const img = Object.create(proto) as { data: number };
    expect(quietImageData(img)).toBe(img);
    expect(img.data).toBe(1);
  });
});

describe('fitWithin', () => {
  it('only scales down, keeping the aspect ratio', () => {
    expect(fitWithin(300, 200, 256)).toEqual({ width: 256, height: 171 });
    expect(fitWithin(40, 20, 256)).toEqual({ width: 40, height: 20 });
  });
});
