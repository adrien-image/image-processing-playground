/**
 * A synthetic segmentation dataset: discs and boxes ("objects") on a background with
 * uneven lighting, noise and thin bright lines ("clutter"). The task is to label every
 * pixel object / not object.
 *
 * It is built so that a global threshold fails: the lighting makes parts of the
 * background brighter than some objects, and the clutter lines are as bright as
 * objects but thin. A network has to use *shape* (filled blobs vs thin lines), which
 * is exactly what convolutions are good at.
 */
import { gaussianRandom } from '../../lib/random';

export interface SegSample {
  /** size × size values in [0, 1] */
  image: Float32Array;
  /** size × size, 1 = object */
  mask: Uint8Array;
}

export interface Difficulty {
  /** How much brighter one side of the image is than the other (0–1). */
  lighting: number;
  /** Standard deviation of the pixel noise. */
  noise: number;
  /** Number of clutter lines. */
  clutter: number;
}

export const DEFAULT_DIFFICULTY: Difficulty = { lighting: 0.45, noise: 0.06, clutter: 3 };

export function makeSample(size: number, random: () => number, { lighting, noise, clutter }: Difficulty = DEFAULT_DIFFICULTY): SegSample {
  const image = new Float32Array(size * size);
  const mask = new Uint8Array(size * size);

  // Lighting: a linear ramp in a random direction.
  const angle = random() * Math.PI * 2;
  const gx = Math.cos(angle), gy = Math.sin(angle);
  const base = 0.1 + random() * 0.15;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const t = ((x / size - 0.5) * gx + (y / size - 0.5) * gy) + 0.5; // 0…1 across the image
      image[y * size + x] = base + lighting * t;
    }
  }

  // Objects: 1–3 discs or boxes, a little brighter than the background around them.
  const objects = 1 + Math.floor(random() * 3);
  for (let o = 0; o < objects; o++) {
    const r = size * (0.1 + random() * 0.12);
    const cx = r + random() * (size - 2 * r), cy = r + random() * (size - 2 * r);
    const disc = random() < 0.5;
    const contrast = 0.25 + random() * 0.2;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
        const inside = disc ? dx * dx + dy * dy <= r * r : Math.abs(dx) <= r * 0.9 && Math.abs(dy) <= r * 0.7;
        if (inside) {
          mask[y * size + x] = 1;
          image[y * size + x] += contrast;
        }
      }
    }
  }

  // Clutter: thin bright lines that are not objects.
  for (let l = 0; l < clutter; l++) {
    const x0 = random() * size, y0 = random() * size;
    const a = random() * Math.PI;
    const len = size * (0.4 + random() * 0.5);
    for (let s = 0; s < len; s += 0.5) {
      const x = Math.floor(x0 + Math.cos(a) * s), y = Math.floor(y0 + Math.sin(a) * s);
      if (x >= 0 && y >= 0 && x < size && y < size && !mask[y * size + x]) image[y * size + x] += 0.35;
    }
  }

  for (let i = 0; i < image.length; i++) image[i] = Math.min(1, Math.max(0, image[i] + noise * gaussianRandom(random)));
  return { image, mask };
}

/** `count` samples packed into flat arrays (for training batches). */
export function makeDataset(count: number, size: number, random: () => number, difficulty?: Difficulty) {
  const images = new Float32Array(count * size * size);
  const masks = new Float32Array(count * size * size);
  for (let n = 0; n < count; n++) {
    const s = makeSample(size, random, difficulty);
    images.set(s.image, n * size * size);
    masks.set(s.mask, n * size * size);
  }
  return { images, masks, count };
}

/** Intersection over union of a predicted mask (probabilities ≥ 0.5 count as object) and the truth. */
export function maskIoU(predicted: ArrayLike<number>, truth: ArrayLike<number>): number {
  let inter = 0, union = 0;
  for (let i = 0; i < truth.length; i++) {
    const p = predicted[i] >= 0.5, t = truth[i] >= 0.5;
    if (p && t) inter++;
    if (p || t) union++;
  }
  return union === 0 ? 1 : inter / union;
}
