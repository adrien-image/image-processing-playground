/** Colour maps shared by sections (no DOM, no TensorFlow.js). */

const NEGATIVE: readonly [number, number, number] = [70, 140, 255];
const POSITIVE: readonly [number, number, number] = [255, 70, 50];

/**
 * Diverging colour map for signed values t in [−1, 1] (clamped):
 * −1 → blue, 0 → black, +1 → red, fading linearly in between.
 */
export function divergingColor(t: number): [number, number, number] {
  const c = Number.isFinite(t) ? Math.max(-1, Math.min(1, t)) : 0;
  const [r, g, b] = c < 0 ? NEGATIVE : POSITIVE;
  const a = Math.abs(c);
  return [r * a, g * a, b * a];
}
