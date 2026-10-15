/** Synthetic demo images and blend masks for the Pyramids section (pure: no DOM). */
import { createPlane, type Plane } from './plane';

/**
 * Zone plate: I = 127.5 + 127.5 · cos(k · r²), r = distance from the centre.
 * The local frequency grows linearly with the distance: (k·(x − c)/π, k·(y − c)/π)
 * cycles/pixel along x and y. k is chosen so its length, the ring frequency k·r/π, reaches
 * 0.5 cycles/pixel at the corners. The pixel grid limits each axis separately (|fx|, |fy| ≤ 0.5),
 * and per axis the zone plate peaks at ≈ 0.35 (along the edges): the original is free of
 * aliasing, but halving (new limit 0.25) or more without a blur already produces moiré.
 */
export function zonePlate(size: number): Plane {
  const c = (size - 1) / 2;
  const k = Math.PI / (2 * Math.SQRT2 * (size / 2));
  const out = createPlane(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) out.data[y * size + x] = 127.5 + 127.5 * Math.cos(k * ((x - c) ** 2 + (y - c) ** 2));
  }
  return out;
}

type RGB = readonly [number, number, number];

const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

function paint(width: number, height: number, colour: (x: number, y: number) => RGB): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = colour(x, y), i = (y * width + x) * 4;
      out[i] = r; out[i + 1] = g; out[i + 2] = b; out[i + 3] = 255;
    }
  }
  return out;
}

/** Warm scene (RGBA): purple→orange→yellow sky with thin cloud bands, a sun disc, striped sea. */
export function sunsetScene(width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const horizon = 0.62 * height;
  const sun = { x: 0.5 * width, y: 0.52 * height, r: 0.14 * Math.min(width, height) };
  return paint(width, height, (x, y) => {
    if (y < horizon) {
      const t = y / horizon;
      let c = t < 0.5 ? mix([88, 24, 72], [232, 84, 44], t / 0.5) : mix([232, 84, 44], [255, 178, 70], (t - 0.5) / 0.5);
      if (Math.hypot(x - sun.x, y - sun.y) < sun.r) c = [255, 232, 130];
      // Thin dark cloud bands.
      if (Math.sin(y * 0.55) > 0.93 && Math.sin(x * 0.02 + y * 0.1) > -0.2) c = mix(c, [70, 20, 50], 0.6);
      return c;
    }
    const t = (y - horizon) / (height - horizon);
    const water = mix([150, 50, 40], [50, 14, 30], t);
    // Horizontal orange stripes: the sun's reflection, strongest below the sun.
    const near = Math.exp(-(((x - sun.x) / (0.25 * width)) ** 2));
    return Math.sin(y * 1.3) > 0.4 ? mix(water, [255, 160, 60], 0.35 + 0.55 * near) : water;
  });
}

/** Cool scene (RGBA): blue sky with a white disc, a green forest ridge, blue-green sinusoidal waves. */
export function seaScene(width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const moon = { x: 0.3 * width, y: 0.2 * height, r: 0.09 * Math.min(width, height) };
  return paint(width, height, (x, y) => {
    const u = x / width, v = y / height;
    const ridge = 0.42 + 0.05 * Math.sin(2 * Math.PI * u * 2.6) + 0.02 * Math.sin(2 * Math.PI * u * 9);
    if (v < ridge) {
      if (Math.hypot(x - moon.x, y - moon.y) < moon.r) return [250, 250, 255];
      return mix([70, 130, 210], [175, 215, 245], v / ridge);
    }
    if (v < 0.6) {
      const leaves = 0.5 + 0.5 * Math.sin(x * 0.9) * Math.sin(y * 0.7);
      return mix([20, 70, 40], [50, 120, 60], leaves);
    }
    const wave = Math.sin(2 * Math.PI * (y / 9 + 0.8 * Math.sin(2 * Math.PI * x / 60)));
    const swell = Math.sin(2 * Math.PI * (x / 37 + y / 53));
    return [20 + 15 * wave, 120 + 45 * wave + 20 * swell, 170 + 50 * wave - 15 * swell];
  });
}

export type MaskShape = 'vertical' | 'horizontal' | 'disc';

/** Binary blend mask (1 = take image A): left half, top half, or a centred disc. */
export function makeMask(shape: MaskShape, width: number, height: number): Plane {
  const out = createPlane(width, height);
  const r = 0.32 * Math.min(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const inside = shape === 'vertical' ? x < width / 2
        : shape === 'horizontal' ? y < height / 2
          : Math.hypot(x + 0.5 - width / 2, y + 0.5 - height / 2) < r;
      out.data[y * width + x] = inside ? 1 : 0;
    }
  }
  return out;
}
