import { rgbImage } from '../../lib/image';
import { hotColor } from '../../image-processing/hough/hough';
import { divergingColor } from '../../lib/colormap';

/** Values in [0, 1] (clamped) as a grayscale image. */
export function floatImage(values: ArrayLike<number>, width: number, height: number): ImageData {
  return rgbImage(width, height, i => {
    const v = Math.round(255 * Math.min(1, Math.max(0, values[i])));
    return [v, v, v];
  });
}

/** Signed values as blue (negative) – black – red (positive), scaled so |max| is full colour. */
export function signedImage(values: ArrayLike<number>, width: number, height: number, scale = maxAbs(values)): ImageData {
  const s = scale || 1;
  return rgbImage(width, height, i => divergingColor(values[i] / s));
}

/** Non-negative values as a heat map (black → red → yellow), scaled so the maximum is bright. */
export function heatImage(values: ArrayLike<number>, width: number, height: number, scale = maxAbs(values)): ImageData {
  const s = scale || 1;
  return rgbImage(width, height, i => hotColor(Math.max(0, values[i]) / s));
}

/** A heat map with the input (0–1) laid over it as a light veil, so the digit stays visible. */
export function heatOverInput(values: ArrayLike<number>, input: ArrayLike<number>, width: number, height: number): ImageData {
  const s = maxAbs(values) || 1;
  return rgbImage(width, height, i => {
    const [r, g, b] = hotColor(Math.max(0, values[i]) / s);
    const veil = 0.35 * Math.min(1, Math.max(0, input[i]));
    return [r + (255 - r) * veil, g + (255 - g) * veil, b + (255 - b) * veil];
  });
}

export function maxAbs(values: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) m = Math.max(m, Math.abs(values[i]));
  return m;
}

/** Ten distinct colours, one per digit class (scatter plots, legends). */
export const CLASS_COLORS = [
  '#f87171', '#fb923c', '#facc15', '#a3e635', '#34d399', '#22d3ee', '#60a5fa', '#a78bfa', '#f472b6', '#e5e7eb',
] as const;
