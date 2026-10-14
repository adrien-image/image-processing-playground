/**
 * Data augmentation: random transformations that produce new training samples
 * from an existing image. Each transform takes a canvas and returns a new canvas of
 * the same size; the random parameters are drawn up front so labels show the
 * values that were actually applied.
 */

import { createCanvas } from '../lib/canvas';

type Transform = (src: HTMLCanvasElement) => HTMLCanvasElement;

/** New canvas the size of `src`, filled black, with `draw` applied. */
function redraw(src: HTMLCanvasElement, draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void): HTMLCanvasElement {
  const { canvas, ctx } = createCanvas(src.width, src.height);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, src.width, src.height);
  draw(ctx, src.width, src.height);
  return canvas;
}

/** Map every RGB value through `fn` (alpha untouched). */
function mapPixels(src: HTMLCanvasElement, fn: (v: number, i: number) => number): HTMLCanvasElement {
  return redraw(src, (ctx, w, h) => {
    ctx.drawImage(src, 0, 0);
    const img = ctx.getImageData(0, 0, w, h);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = fn(d[i], i);
      d[i + 1] = fn(d[i + 1], i);
      d[i + 2] = fn(d[i + 2], i);
    }
    ctx.putImageData(img, 0, 0);
  });
}

/** Rotate about the centre by `degrees`; size and scale are kept, uncovered corners are black. */
export const rotate = (degrees: number): Transform => src =>
  redraw(src, (ctx, w, h) => {
    ctx.translate(w / 2, h / 2);
    ctx.rotate((degrees * Math.PI) / 180);
    ctx.drawImage(src, -w / 2, -h / 2);
  });

export const flip = (direction: 'horizontal' | 'vertical'): Transform => src =>
  redraw(src, (ctx, w, h) => {
    if (direction === 'horizontal') { ctx.translate(w, 0); ctx.scale(-1, 1); }
    else { ctx.translate(0, h); ctx.scale(1, -1); }
    ctx.drawImage(src, 0, 0);
  });

/** Keep a `scale`-sized window at relative offset (fx, fy) ∈ [0, 1]², resized back to full size. */
export const crop = (scale: number, fx: number, fy: number): Transform => src =>
  redraw(src, (ctx, w, h) => {
    const cw = Math.round(w * scale), ch = Math.round(h * scale);
    ctx.drawImage(src, Math.round((w - cw) * fx), Math.round((h - ch) * fy), cw, ch, 0, 0, w, h);
  });

/** Horizontal shear: x' = x + factor·(y − h/2). */
export const shear = (factor: number): Transform => src =>
  redraw(src, (ctx, _w, h) => {
    ctx.transform(1, 0, factor, 1, -factor * (h / 2), 0);
    ctx.drawImage(src, 0, 0);
  });

/** Multiply every channel: > 1 brightens, < 1 darkens. */
export const brightness = (factor: number): Transform => src => mapPixels(src, v => v * factor);

/** Scale distances from mid-gray: > 1 increases contrast. */
export const contrast = (factor: number): Transform => src => mapPixels(src, v => 128 + (v - 128) * factor);

/** Add the same Gaussian noise sample to R, G and B of each pixel. */
export const gaussianNoise = (std: number, random: () => number): Transform => src => {
  let pixel = -1, noise = 0;
  return mapPixels(src, (v, i) => {
    if (i !== pixel) {
      pixel = i;
      // Box–Muller transform
      const u = 1 - random(), w = random();
      noise = std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
    }
    return v + noise;
  });
};

/** Set a `density` fraction of pixels to pure black or white. */
export const saltPepper = (density: number, random: () => number): Transform => src => {
  let pixel = -1, value = -1;
  return mapPixels(src, (v, i) => {
    if (i !== pixel) {
      pixel = i;
      const r = random();
      value = r < density / 2 ? 0 : r < density ? 255 : -1;
    }
    return value < 0 ? v : value;
  });
};

export const AUGMENTATIONS = ['rotate', 'flip', 'crop', 'brightness', 'contrast', 'gaussianNoise', 'saltPepper', 'shear'] as const;
export type AugmentationType = (typeof AUGMENTATIONS)[number];

export interface AugmentStep {
  type: AugmentationType;
  /** Human-readable description with the parameters actually used */
  label: string;
  apply: Transform;
}

/** Draw random parameters for each enabled augmentation, in pipeline order. */
export function sampleAugmentations(enabled: Record<AugmentationType, boolean>, random: () => number): AugmentStep[] {
  const between = (lo: number, hi: number) => lo + random() * (hi - lo);
  const round2 = (v: number) => Math.round(v * 100) / 100;
  const steps: AugmentStep[] = [];
  for (const type of AUGMENTATIONS) {
    if (!enabled[type]) continue;
    switch (type) {
      case 'rotate': {
        const deg = Math.round(between(-30, 30));
        steps.push({ type, label: `Rotate ${deg > 0 ? '+' : ''}${deg}°`, apply: rotate(deg) });
        break;
      }
      case 'flip': {
        const dir = random() < 0.5 ? 'horizontal' : 'vertical';
        steps.push({ type, label: `Flip ${dir}`, apply: flip(dir) });
        break;
      }
      case 'crop': {
        const s = round2(between(0.7, 0.9));
        steps.push({ type, label: `Crop ${Math.round(s * 100)}%`, apply: crop(s, random(), random()) });
        break;
      }
      case 'brightness': {
        const b = round2(between(0.5, 1.5));
        steps.push({ type, label: `Brightness ×${b.toFixed(2)}`, apply: brightness(b) });
        break;
      }
      case 'contrast': {
        const c = round2(between(0.5, 2));
        steps.push({ type, label: `Contrast ×${c.toFixed(2)}`, apply: contrast(c) });
        break;
      }
      case 'gaussianNoise': {
        const std = Math.round(between(5, 35));
        steps.push({ type, label: `Gaussian noise σ=${std}`, apply: gaussianNoise(std, random) });
        break;
      }
      case 'saltPepper': {
        const d = round2(between(0.02, 0.17));
        steps.push({ type, label: `S&P ${Math.round(d * 100)}%`, apply: saltPepper(d, random) });
        break;
      }
      case 'shear': {
        const s = round2(between(-0.3, 0.3));
        steps.push({ type, label: `Shear ${s > 0 ? '+' : ''}${s.toFixed(2)}`, apply: shear(s) });
        break;
      }
    }
  }
  return steps;
}
