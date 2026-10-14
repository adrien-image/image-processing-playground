/** Colour-space conversions and per-pixel colour adjustments. */

/** RGB (0–255) → HSV with h, s, v in [0, 1]. */
export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const d = mx - mn;
  let h = 0;
  if (d !== 0) {
    if (mx === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (mx === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  const s = mx === 0 ? 0 : d / mx;
  return [h, s, mx];
}

/** HSV (each in [0, 1]) → RGB (0–255). */
export function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r: number, g: number, b: number;
  switch (((i % 6) + 6) % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  return [r * 255, g * 255, b * 255];
}

export interface ColorAdjustments {
  /** Degrees to rotate the hue wheel (0–359). */
  hueShift: number;
  /** Saturation multiplier (0 = grayscale). */
  satScale: number;
  /** Value (brightness) multiplier. */
  valScale: number;
  /** Gamma: output = input^(1/γ); γ < 1 darkens mid-tones, γ > 1 brightens them. */
  gamma: number;
}

export const NO_ADJUSTMENT: ColorAdjustments = { hueShift: 0, satScale: 1, valScale: 1, gamma: 1 };

/** Apply HSV adjustments, then gamma, to RGBA data. Returns a new buffer. */
export function adjustColors(data: Uint8ClampedArray, adj: ColorAdjustments): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(data);
  const { hueShift, satScale, valScale, gamma } = adj;
  const hsvChange = hueShift !== 0 || satScale !== 1 || valScale !== 1;
  const invGamma = 1 / gamma;
  for (let i = 0; i < out.length; i += 4) {
    let r = out[i], g = out[i + 1], b = out[i + 2];
    if (hsvChange) {
      const [h, s, v] = rgbToHsv(r, g, b);
      [r, g, b] = hsvToRgb((h + hueShift / 360) % 1, Math.min(1, s * satScale), Math.min(1, v * valScale));
    }
    if (gamma !== 1) {
      r = Math.pow(r / 255, invGamma) * 255;
      g = Math.pow(g / 255, invGamma) * 255;
      b = Math.pow(b / 255, invGamma) * 255;
    }
    out[i] = r; out[i + 1] = g; out[i + 2] = b;
  }
  return out;
}

export type Channel = 'r' | 'g' | 'b' | 'h' | 's' | 'v';

/**
 * Visualise one channel. R/G/B/S/V become grayscale maps; H is shown as the pure
 * hue (full saturation and value) so it can be read as a colour.
 */
export function extractChannel(data: Uint8ClampedArray, channel: Channel): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    let o: [number, number, number];
    if (channel === 'r') o = [r, r, r];
    else if (channel === 'g') o = [g, g, g];
    else if (channel === 'b') o = [b, b, b];
    else {
      const [h, s, v] = rgbToHsv(r, g, b);
      if (channel === 'h') o = hsvToRgb(h, 1, 1);
      else { const x = (channel === 's' ? s : v) * 255; o = [x, x, x]; }
    }
    out[i] = o[0]; out[i + 1] = o[1]; out[i + 2] = o[2]; out[i + 3] = data[i + 3];
  }
  return out;
}
