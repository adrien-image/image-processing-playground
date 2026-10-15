/**
 * Depth from a stereo pair.
 *
 * Two cameras side by side see the same scene shifted horizontally; near objects shift
 * more than far ones. That shift is the disparity d, and depth = focal length · baseline / d.
 * For each pixel of the left image, block matching tries every disparity 0…maxD and keeps
 * the one whose window in the right image (shifted left by d) looks most alike (smallest
 * sum of absolute differences, SAD).
 *
 *   window size   small windows are noisy (few pixels to compare, ambiguous on flat texture);
 *                 large ones are robust but blur depth edges ("foreground fattening")
 *   occlusions    pixels visible in one image only have no true match; a left–right
 *                 consistency check finds them: match right→left too and keep only pixels
 *                 where both directions agree
 */
import { gaussianRandom } from '../../lib/random';

export interface StereoLayer {
  /** disparity in pixels: larger = closer to the cameras */
  disparity: number;
  /** does the layer cover left-image pixel (x, y)? */
  contains: (x: number, y: number) => boolean;
  /** its own texture, so layers can be told apart */
  seed: number;
}

/** A random-dot-like texture (blurry blobs on noise), different for every seed. */
function textureAt(x: number, y: number, seed: number): number {
  const s = seed * 12.9898;
  return 128
    + 45 * Math.sin(x * 0.9 + s) * Math.cos(y * 0.7 - s)
    + 35 * Math.sin((x * 0.31 + y * 0.53) + s * 1.7)
    + 25 * Math.cos(x * 1.7 - y * 1.3 + s * 0.3);
}

export interface StereoPair {
  left: Float64Array;
  right: Float64Array;
  /** true disparity of every left pixel */
  disparity: Float64Array;
  /** left pixels hidden in the right image by a nearer layer */
  occluded: Uint8Array;
  width: number;
  height: number;
}

/**
 * Render a pair: the left camera sees each layer at its own position, the right camera sees
 * every layer shifted left by its disparity; nearer layers (larger disparity) are drawn on top.
 * Textures are painted on the objects, so they move with them.
 */
export function renderStereo(width: number, height: number, layers: StereoLayer[], noise: number, random: () => number): StereoPair {
  const order = [...layers].sort((a, b) => a.disparity - b.disparity); // far to near
  const left = new Float64Array(width * height);
  const right = new Float64Array(width * height);
  const disparity = new Float64Array(width * height);
  const occluded = new Uint8Array(width * height);
  const frontAt = (x: number, y: number) => {
    let front: StereoLayer | null = null;
    for (const l of order) if (l.contains(x, y)) front = l;
    return front;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const l = frontAt(x, y);
      if (l) {
        left[y * width + x] = textureAt(x, y, l.seed);
        disparity[y * width + x] = l.disparity;
      }
      // Right image: the nearest layer whose left-image position x + d covers this pixel.
      let r: StereoLayer | null = null;
      for (const layer of order) if (layer.contains(x + layer.disparity, y)) r = layer;
      if (r) right[y * width + x] = textureAt(x + r.disparity, y, r.seed);
      if (noise > 0) {
        left[y * width + x] += noise * gaussianRandom(random);
        right[y * width + x] += noise * gaussianRandom(random);
      }
    }
  }
  // A left pixel is occluded if, where it lands in the right image, a nearer layer is in front.
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const d = disparity[y * width + x];
    const xr = x - d;
    if (xr < 0) { occluded[y * width + x] = 1; continue; }
    for (const layer of order) {
      if (layer.disparity > d && layer.contains(xr + layer.disparity, y)) { occluded[y * width + x] = 1; break; }
    }
  }
  return { left, right, disparity, occluded, width, height };
}

/** The demo scene: a textured wall, a box, a disc and a thin pole at different depths. */
export function demoLayers(width: number, height: number): StereoLayer[] {
  return [
    { disparity: 4, seed: 1, contains: () => true },
    { disparity: 10, seed: 2, contains: (x, y) => x > width * 0.12 && x < width * 0.45 && y > height * 0.2 && y < height * 0.8 },
    { disparity: 16, seed: 3, contains: (x, y) => (x - width * 0.68) ** 2 + (y - height * 0.45) ** 2 < (height * 0.22) ** 2 },
    { disparity: 22, seed: 4, contains: x => x > width * 0.86 && x < width * 0.9 },
  ];
}

/**
 * Block matching. Builds the cost of every disparity as a "cost volume": per-pixel
 * |L(x, y) − R(x − d, y)|, summed over the window with an integral image, so the window
 * size doesn't change the cost. Returns the disparity with the smallest cost per pixel
 * (−1 where no disparity fits inside the image).
 *
 * `fromRight` matches the other way (each right pixel against left pixels at x + d), for
 * the left–right check.
 */
export function blockMatch(
  left: Float64Array, right: Float64Array, width: number, height: number, maxDisparity: number, radius: number, fromRight = false,
): Int16Array {
  const best = new Float64Array(width * height).fill(Infinity);
  const result = new Int16Array(width * height).fill(-1);
  const diff = new Float64Array(width * height);
  const integral = new Float64Array((width + 1) * (height + 1));
  const W = width + 1;
  for (let d = 0; d <= maxDisparity; d++) {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const xo = fromRight ? x + d : x - d;
      diff[y * width + x] = xo < 0 || xo >= width ? 255
        : fromRight ? Math.abs(right[y * width + x] - left[y * width + xo]) : Math.abs(left[y * width + x] - right[y * width + xo]);
    }
    for (let y = 0; y < height; y++) {
      let row = 0;
      for (let x = 0; x < width; x++) {
        row += diff[y * width + x];
        integral[(y + 1) * W + x + 1] = integral[y * W + x + 1] + row;
      }
    }
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(width - 1, x + radius);
      const y0 = Math.max(0, y - radius), y1 = Math.min(height - 1, y + radius);
      const sum = integral[(y1 + 1) * W + x1 + 1] - integral[y0 * W + x1 + 1] - integral[(y1 + 1) * W + x0] + integral[y0 * W + x0];
      const cost = sum / ((x1 - x0 + 1) * (y1 - y0 + 1));
      if (cost < best[y * width + x]) { best[y * width + x] = cost; result[y * width + x] = d; }
    }
  }
  return result;
}

/** Keep a left disparity only if the right image's match points back to (within 1 px of) it; −1 otherwise. */
export function leftRightCheck(dLeft: Int16Array, dRight: Int16Array, width: number, height: number): Int16Array {
  const out = new Int16Array(dLeft.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const d = dLeft[y * width + x];
    const xr = x - d;
    out[y * width + x] = d >= 0 && xr >= 0 && Math.abs(dRight[y * width + xr] - d) <= 1 ? d : -1;
  }
  return out;
}

export interface StereoScore {
  /** share of visible, matched pixels whose disparity is off by more than 1 */
  bad: number;
  /** share of pixels left without a disparity (rejected) */
  invalid: number;
  /** share of occluded pixels that were correctly rejected */
  occlusionsCaught: number;
}

export function score(estimate: Int16Array, truth: StereoPair, margin: number): StereoScore {
  let bad = 0, matched = 0, invalid = 0, total = 0, occ = 0, caught = 0;
  const { width, height } = truth;
  for (let y = 0; y < height; y++) for (let x = margin; x < width; x++) {
    const i = y * width + x;
    total++;
    if (estimate[i] < 0) invalid++;
    if (truth.occluded[i]) { occ++; if (estimate[i] < 0) caught++; continue; }
    if (estimate[i] < 0) continue;
    matched++;
    if (Math.abs(estimate[i] - truth.disparity[i]) > 1) bad++;
  }
  return { bad: matched ? bad / matched : 0, invalid: invalid / total, occlusionsCaught: occ ? caught / occ : 1 };
}
