import { makeCanvas } from '../lib/demoImages';
import type { Similarity } from './transform';

export const SCENE_WIDTH = 220;
export const SCENE_HEIGHT = 200;

/**
 * How image B is produced from the scene, in canvas terms: rotate by `rotate` radians
 * and scale by `scale` around the image centre, then shift by (tx, ty).
 */
export interface SceneTransform {
  rotate: number;
  scale: number;
  tx: number;
  ty: number;
}

export const DEFAULT_TRANSFORM: SceneTransform = { rotate: 0.4, scale: 0.85, tx: 8, ty: -6 };

/** Draw the synthetic scene, optionally transformed. */
export function drawScene(transform?: SceneTransform): HTMLCanvasElement {
  return makeCanvas(SCENE_WIDTH, SCENE_HEIGHT, ctx => {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, SCENE_WIDTH, SCENE_HEIGHT);
    if (transform) {
      ctx.translate(SCENE_WIDTH / 2 + transform.tx, SCENE_HEIGHT / 2 + transform.ty);
      ctx.rotate(transform.rotate);
      ctx.scale(transform.scale, transform.scale);
      ctx.translate(-SCENE_WIDTH / 2, -SCENE_HEIGHT / 2);
    }
    ctx.fillStyle = '#ddd';
    ctx.fillRect(40, 40, 120, 80);
    ctx.fillStyle = '#888';
    ctx.fillRect(60, 60, 60, 60);
    ctx.fillStyle = '#aaa';
    ctx.beginPath(); ctx.arc(180, 60, 30, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#666';
    ctx.beginPath(); ctx.arc(180, 120, 20, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.strokeRect(31, 151, 138, 48);
    ctx.fillStyle = '#999'; ctx.fillRect(50, 160, 30, 30);
  });
}

/**
 * The transform from image-A pixel coordinates to image-B pixel coordinates.
 * Canvas applies translate(c + t) · rotate(θ) · scale(s) · translate(−c), which expands to
 * x' = s·R·x + (c − s·R·c + t) with c the image centre.
 */
export function groundTruth(t: SceneTransform): Similarity {
  const cx = SCENE_WIDTH / 2, cy = SCENE_HEIGHT / 2;
  const a = t.scale * Math.cos(t.rotate), b = t.scale * Math.sin(t.rotate);
  return {
    angle: t.rotate,
    scale: t.scale,
    tx: cx - (a * cx - b * cy) + t.tx,
    ty: cy - (b * cx + a * cy) + t.ty,
  };
}

/** A random but reasonable transform for the "new transform" button. */
export function randomTransform(random: () => number = Math.random): SceneTransform {
  const r = (lo: number, hi: number) => lo + random() * (hi - lo);
  return {
    rotate: Math.round(r(-1, 1) * 100) / 100,
    scale: Math.round(r(0.75, 1.1) * 100) / 100,
    tx: Math.round(r(-12, 12)),
    ty: Math.round(r(-12, 12)),
  };
}
