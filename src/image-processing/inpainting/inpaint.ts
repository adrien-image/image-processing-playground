/**
 * Inpainting: fill a hole in an image from its surroundings.
 *
 * Diffusion: make every missing pixel the average of its 4 neighbours (Laplace's equation
 * ∇²u = 0 inside the hole, with the known pixels as boundary values), solved by repeated
 * Gauss–Seidel sweeps. Like heat spreading in from the border, it gives the smoothest
 * possible fill: perfect for a gradient, a blur across texture and edges.
 *
 * Exemplar-based (Criminisi, Pérez & Toyama 2004): repeatedly pick a patch on the hole's
 * border, find the most similar fully-known patch elsewhere in the image (SSD over the
 * pixels already known), and copy its pixels into the hole. Which border patch goes first
 * matters: priority = confidence × data term, so patches surrounded by many known pixels and
 * patches where a strong edge runs into the hole are filled first, which carries lines and
 * texture inwards.
 */

export interface RgbaImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** Diffusion fill (Laplace) with `iterations` Gauss–Seidel sweeps; holes start at the mean border colour. */
export function diffusionInpaint(img: RgbaImage, mask: Uint8Array, iterations: number): Uint8ClampedArray<ArrayBuffer> {
  const { width, height } = img;
  const n = width * height;
  const hole: number[] = [];
  for (let i = 0; i < n; i++) if (mask[i]) hole.push(i);
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[i * 3 + c] = img.data[i * 4 + c];
  // Start from the average colour of the known pixels touching the hole.
  const start = [0, 0, 0];
  let count = 0;
  for (const i of hole) {
    for (const j of neighbours(i, width, height)) {
      if (mask[j]) continue;
      for (let c = 0; c < 3; c++) start[c] += img.data[j * 4 + c];
      count++;
    }
  }
  for (const i of hole) for (let c = 0; c < 3; c++) out[i * 3 + c] = count ? start[c] / count : 128;
  for (let it = 0; it < iterations; it++) {
    for (const i of hole) {
      const nb = neighbours(i, width, height);
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (const j of nb) s += out[j * 3 + c];
        out[i * 3 + c] = s / nb.length;
      }
    }
  }
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) rgba[i * 4 + c] = out[i * 3 + c];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function neighbours(i: number, width: number, height: number): number[] {
  const x = i % width, y = (i - x) / width, out: number[] = [];
  if (x > 0) out.push(i - 1);
  if (x < width - 1) out.push(i + 1);
  if (y > 0) out.push(i - width);
  if (y < height - 1) out.push(i + width);
  return out;
}

export interface ExemplarResult {
  /** The filled image (RGBA). */
  data: Uint8ClampedArray<ArrayBuffer>;
  /** For each pixel, the step at which it was filled (0 = known from the start). */
  filledAt: Int32Array;
  /** Per step: the target patch centre and the source patch centre it copied from. */
  steps: { target: [number, number]; source: [number, number] }[];
}

/**
 * Criminisi exemplar-based inpainting.
 * @param radius patch half-size (patch is (2r+1)²)
 * @param searchRadius candidate source patches are looked for within this distance of the target
 * @param usePriority false fills border patches in plain scan order ("onion peel"), to compare
 */
export function exemplarInpaint(img: RgbaImage, mask: Uint8Array, radius = 4, searchRadius = 40, usePriority = true): ExemplarResult {
  const { width, height } = img;
  const n = width * height;
  const data = new Uint8ClampedArray(img.data);
  const known = Uint8Array.from(mask, m => (m ? 0 : 1));
  const confidence = Float64Array.from(known);
  const filledAt = new Int32Array(n);
  const gray = new Float64Array(n);
  const updateGray = (i: number) => { gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]; };
  for (let i = 0; i < n; i++) updateGray(i);
  const steps: ExemplarResult['steps'] = [];
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height;

  // Candidate sources: centres whose whole patch lies inside the image and is fully known.
  const isSource = (cx: number, cy: number) => {
    if (cx < radius || cy < radius || cx >= width - radius || cy >= height - radius) return false;
    for (let y = cy - radius; y <= cy + radius; y++) for (let x = cx - radius; x <= cx + radius; x++) if (mask[y * width + x]) return false;
    return true;
  };
  const sourceOk = new Uint8Array(n);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) sourceOk[y * width + x] = isSource(x, y) ? 1 : 0;

  for (let step = 1; step < n; step++) {
    // 1. The fill front: unknown pixels with a known neighbour.
    let best = -1, bestPriority = -Infinity;
    for (let i = 0; i < n; i++) {
      if (known[i]) continue;
      const nb = neighbours(i, width, height);
      if (!nb.some(j => known[j])) continue;
      if (!usePriority) { best = i; break; }
      const x = i % width, y = (i - x) / width;
      // Confidence: how much of the patch is (reliably) known.
      let c = 0, area = 0;
      for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
        if (!inside(x + dx, y + dy)) continue;
        area++;
        c += confidence[(y + dy) * width + x + dx];
      }
      c /= area;
      // Data term: how strongly an edge (isophote) runs into the front, |∇I⊥ · n|. The gradient is
      // the strongest one among known neighbours whose own neighbours are known; n is the front's normal.
      const at = (xx: number, yy: number) => (inside(xx, yy) && known[yy * width + xx] ? gray[yy * width + xx] : NaN);
      let gx = 0, gy = 0, nx = 0, ny = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (!inside(xx, yy)) continue;
        if (!known[yy * width + xx]) { nx -= dx; ny -= dy; continue; }
        const ex = (at(xx + 1, yy) - at(xx - 1, yy)) / 2, ey = (at(xx, yy + 1) - at(xx, yy - 1)) / 2;
        if (Number.isFinite(ex) && Number.isFinite(ey) && ex * ex + ey * ey > gx * gx + gy * gy) { gx = ex; gy = ey; }
      }
      const nlen = Math.hypot(nx, ny) || 1;
      const d = Math.abs(-gy * (nx / nlen) + gx * (ny / nlen)) / 255 + 0.001;   // isophote = gradient rotated 90°
      const priority = c * d;
      if (priority > bestPriority) { bestPriority = priority; best = i; }
    }
    if (best < 0) break;
    const tx = best % width, ty = (best - tx) / width;

    // 2. Best source patch: smallest SSD over the target's known pixels.
    let bestSource = -1, bestCost = Infinity;
    for (let sy = Math.max(radius, ty - searchRadius); sy <= Math.min(height - 1 - radius, ty + searchRadius); sy++) {
      for (let sx = Math.max(radius, tx - searchRadius); sx <= Math.min(width - 1 - radius, tx + searchRadius); sx++) {
        if (!sourceOk[sy * width + sx]) continue;
        let cost = 0;
        for (let dy = -radius; dy <= radius && cost < bestCost; dy++) for (let dx = -radius; dx <= radius; dx++) {
          const x = tx + dx, y = ty + dy;
          if (!inside(x, y)) continue;
          const t = y * width + x;
          if (!known[t]) continue;
          const s = (sy + dy) * width + sx + dx;
          const dr = data[t * 4] - data[s * 4], dg = data[t * 4 + 1] - data[s * 4 + 1], db = data[t * 4 + 2] - data[s * 4 + 2];
          cost += dr * dr + dg * dg + db * db;
        }
        if (cost < bestCost) { bestCost = cost; bestSource = sy * width + sx; }
      }
    }
    if (bestSource < 0) {                                   // nothing to copy from: fall back to the mean of known neighbours
      const nb = neighbours(best, width, height).filter(j => known[j]);
      for (let c = 0; c < 3; c++) data[best * 4 + c] = nb.reduce((s, j) => s + data[j * 4 + c], 0) / nb.length;
      known[best] = 1; filledAt[best] = step; updateGray(best);
      continue;
    }
    const sx = bestSource % width, sy = (bestSource - sx) / width;

    // 3. Copy the unknown pixels of the target patch; they inherit the patch's confidence.
    let c = 0, area = 0;
    for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
      if (!inside(tx + dx, ty + dy)) continue;
      area++;
      c += confidence[(ty + dy) * width + tx + dx];
    }
    c /= area;
    for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
      const x = tx + dx, y = ty + dy;
      if (!inside(x, y)) continue;
      const t = y * width + x;
      if (known[t]) continue;
      const s = (sy + dy) * width + sx + dx;
      data[t * 4] = data[s * 4]; data[t * 4 + 1] = data[s * 4 + 1]; data[t * 4 + 2] = data[s * 4 + 2];
      known[t] = 1; confidence[t] = c; filledAt[t] = step; updateGray(t);
    }
    steps.push({ target: [tx, ty], source: [sx, sy] });
  }
  return { data, filledAt, steps };
}

/** Root-mean-square colour error over the masked pixels. */
export function holeError(original: Uint8ClampedArray, filled: Uint8ClampedArray, mask: Uint8Array): number {
  let s = 0, count = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    for (let c = 0; c < 3; c++) s += (original[i * 4 + c] - filled[i * 4 + c]) ** 2;
    count += 3;
  }
  return count ? Math.sqrt(s / count) : 0;
}
