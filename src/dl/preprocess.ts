/**
 * Turn a free-hand drawing into an MNIST-style 28×28 input.
 *
 * MNIST digits were size-normalised to fit a 20×20 box (keeping their aspect ratio)
 * and then centred by their centre of mass in a 28×28 frame. A raw downscale of the
 * drawing canvas doesn't match that — small or off-centre digits look nothing like
 * the training data — so we reproduce the same normalisation.
 */

export const MNIST_SIZE = 28;
const BOX = 20;
const INK = 0.1;

/**
 * @param pixels  - grayscale drawing, values 0 (background) to 1 (ink), row-major
 * @returns 784 values in [0, 1], or null if nothing was drawn
 */
export function toMnistInput(pixels: Float32Array, width: number, height: number): Float32Array | null {
  // 1. Bounding box of the ink
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (pixels[y * width + x] > INK) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;

  // 2. Scale the box so its longer side is 20 px (area-averaged resampling)
  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const scale = BOX / Math.max(bw, bh);
  const tw = Math.max(1, Math.round(bw * scale)), th = Math.max(1, Math.round(bh * scale));
  const scaled = new Float32Array(tw * th);
  const SS = 4; // sub-samples per output pixel per axis
  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      let sum = 0;
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const x = Math.min(maxX, Math.floor(minX + ((tx + (sx + 0.5) / SS) * bw) / tw));
          const y = Math.min(maxY, Math.floor(minY + ((ty + (sy + 0.5) / SS) * bh) / th));
          sum += pixels[y * width + x];
        }
      }
      scaled[ty * tw + tx] = sum / (SS * SS);
    }
  }

  // 3. Centre of mass of the scaled digit
  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const v = scaled[y * tw + x];
      mass += v;
      cx += v * (x + 0.5);
      cy += v * (y + 0.5);
    }
  }
  cx /= mass;
  cy /= mass;

  // 4. Paste it so the centre of mass lands at the centre of the 28×28 frame
  const out = new Float32Array(MNIST_SIZE * MNIST_SIZE);
  const ox = Math.round(MNIST_SIZE / 2 - cx), oy = Math.round(MNIST_SIZE / 2 - cy);
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const X = x + ox, Y = y + oy;
      if (X >= 0 && X < MNIST_SIZE && Y >= 0 && Y < MNIST_SIZE) out[Y * MNIST_SIZE + X] = Math.min(1, scaled[y * tw + x]);
    }
  }
  return out;
}
