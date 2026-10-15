/**
 * Connected-component labelling and region measurements.
 *
 * Input is a binary mask: a Uint8Array in row-major order (index y·width + x) with
 * 1 for foreground and 0 for background. Labelling gives every connected blob of
 * foreground pixels its own number (1, 2, 3, …); `regionProperties` then measures
 * each blob: how big it is, where it is, and what shape it has.
 *
 * Coordinates: pixel (x, y) is the unit square [x, x+1) × [y, y+1), with the origin at
 * the top-left corner of the image. A pixel's centre is therefore (x + 0.5, y + 0.5).
 */

export type Connectivity = 4 | 8;

export interface Labelling {
  /** One label per pixel: 0 = background, 1..count = region number. */
  labels: Int32Array;
  /** Number of regions. */
  count: number;
}

/**
 * Neighbour offsets [dx, dy]. 4-connectivity: pixels that share an edge.
 * 8-connectivity: pixels that share an edge or just a corner (the diagonals).
 */
const NEIGHBOURS_4: readonly (readonly [number, number])[] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const NEIGHBOURS_8: readonly (readonly [number, number])[] = [...NEIGHBOURS_4, [1, 1], [-1, 1], [1, -1], [-1, -1]];

/**
 * Label the connected components of a binary mask by flood fill.
 *
 * Scan the pixels in raster order (left to right, top to bottom). Each time we meet a
 * foreground pixel that has no label yet, it is the first pixel of a new region: give it
 * the next label and flood-fill outwards with an explicit stack (no recursion, so huge
 * regions cannot overflow the call stack). A pixel is labelled the moment it is pushed,
 * so it is never pushed twice and the whole pass is O(width · height).
 *
 * Because new labels are handed out during the raster scan, regions are numbered
 * 1..count in the raster order of their first (top-most, then left-most) pixel.
 */
export function labelComponents(mask: Uint8Array, width: number, height: number, connectivity: Connectivity): Labelling {
  const labels = new Int32Array(width * height);
  const offsets = connectivity === 4 ? NEIGHBOURS_4 : NEIGHBOURS_8;
  const stack = new Int32Array(width * height); // pixels waiting to be visited
  let count = 0;

  for (let start = 0; start < width * height; start++) {
    if (!mask[start] || labels[start]) continue; // background, or already in a region

    count++;
    labels[start] = count;
    let top = 0;
    stack[top++] = start;

    while (top > 0) {
      const i = stack[--top];
      const x = i % width;
      const y = (i - x) / width;
      for (const [dx, dy] of offsets) {
        const nx = x + dx;
        const ny = y + dy;
        // Stay inside the image: (width−1, y) and (0, y+1) are not neighbours.
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const n = ny * width + nx;
        if (mask[n] && !labels[n]) {
          labels[n] = count;
          stack[top++] = n;
        }
      }
    }
  }
  return { labels, count };
}

export interface Region {
  label: number;
  /** Number of pixels. */
  area: number;
  /** Mean of the pixel centres (x + 0.5, y + 0.5), in image pixels from the top-left corner. */
  centroid: { x: number; y: number };
  /** Smallest axis-aligned box holding the region; inclusive pixel indices. */
  bbox: { x0: number; y0: number; x1: number; y1: number };
  /**
   * Number of boundary pixels: region pixels with at least one 4-neighbour (left, right,
   * up, down) that is not in the region, i.e. background or outside the image.
   */
  perimeter: number;
  /**
   * 4πA / P². Equal to 1 for a continuous circle, but only approximate on a pixel grid:
   * the boundary-pixel count above counts a diagonal step as 1 instead of √2, so a digital
   * disk scores about 1.25 (tending to π²/8 ≈ 1.23 as it grows), and a large square tends
   * to π/4 ≈ 0.79 (a 10×10 square still scores ≈ 0.97, a 5×5 one ≈ 1.23).
   * Meaningless for tiny regions (a single pixel scores 4π ≈ 12.6).
   */
  circularity: number;
  /** Mean gray value (0–255) of the region's pixels, or null when no image was given. */
  meanIntensity: number | null;
}

/**
 * Measure every region of a labelling in a single pass over the pixels.
 * Per-label running sums (area, Σx, Σy, Σgray, boundary pixels, box corners) are kept in
 * arrays indexed by label; each label 1..count is expected to occur at least once.
 */
export function regionProperties(
  labels: Int32Array,
  count: number,
  width: number,
  height: number,
  gray?: Float64Array,
): Region[] {
  const area = new Float64Array(count + 1);
  const sumX = new Float64Array(count + 1);
  const sumY = new Float64Array(count + 1);
  const sumGray = new Float64Array(count + 1);
  const boundary = new Float64Array(count + 1);
  const x0 = new Int32Array(count + 1).fill(width);
  const y0 = new Int32Array(count + 1).fill(height);
  const x1 = new Int32Array(count + 1).fill(-1);
  const y1 = new Int32Array(count + 1).fill(-1);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const l = labels[i];
      if (l === 0) continue;

      area[l]++;
      sumX[l] += x + 0.5;
      sumY[l] += y + 0.5;
      if (gray) sumGray[l] += gray[i];
      if (x < x0[l]) x0[l] = x;
      if (x > x1[l]) x1[l] = x;
      if (y < y0[l]) y0[l] = y;
      if (y > y1[l]) y1[l] = y;

      // Boundary pixel: on the image border, or a 4-neighbour belongs to another label.
      const onBoundary = x === 0 || y === 0 || x === width - 1 || y === height - 1
        || labels[i - 1] !== l || labels[i + 1] !== l || labels[i - width] !== l || labels[i + width] !== l;
      if (onBoundary) boundary[l]++;
    }
  }

  const regions: Region[] = [];
  for (let l = 1; l <= count; l++) {
    const a = area[l];
    const p = boundary[l];
    regions.push({
      label: l,
      area: a,
      centroid: { x: sumX[l] / a, y: sumY[l] / a },
      bbox: { x0: x0[l], y0: y0[l], x1: x1[l], y1: y1[l] },
      perimeter: p,
      circularity: p > 0 ? (4 * Math.PI * a) / (p * p) : 0,
      meanIntensity: gray ? sumGray[l] / a : null,
    });
  }
  return regions;
}

export interface FilteredRegions extends Labelling {
  regions: Region[];
}

/**
 * Drop regions with fewer than `minArea` pixels and renumber the rest 1..count, keeping
 * their order (so labels stay in raster order). Dropped pixels become background (0).
 * `regions` must be the output of `regionProperties` for these labels.
 */
export function removeSmall(labels: Int32Array, regions: Region[], minArea: number): FilteredRegions {
  // Lookup table: old label → new label (0 = dropped). Index 0 (background) stays 0.
  const newLabel = new Int32Array(regions.length + 1);
  const kept: Region[] = [];
  for (const r of regions) {
    if (r.area < minArea) continue;
    kept.push({ ...r, label: kept.length + 1 });
    newLabel[r.label] = kept.length;
  }
  return { labels: labels.map(l => newLabel[l]), count: kept.length, regions: kept };
}

/** Golden angle in degrees: successive multiples of it spread hues evenly around the wheel. */
const GOLDEN_ANGLE = 137.508;

/**
 * A distinct colour for each label, for drawing the label map. Hues step by the golden
 * angle, so neighbouring label numbers never get similar colours. Label 0 is black.
 */
export function labelColour(label: number): [number, number, number] {
  if (label <= 0) return [0, 0, 0];
  return hslToRgb((label * GOLDEN_ANGLE) % 360, 0.75, 0.58);
}

/** HSL (h in degrees, s and l in 0–1) → RGB in 0–255. */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s; // chroma
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const [r, g, b] = hp < 1 ? [c, x, 0] : hp < 2 ? [x, c, 0] : hp < 3 ? [0, c, x]
    : hp < 4 ? [0, x, c] : hp < 5 ? [x, 0, c] : [c, 0, x];
  const m = l - c / 2;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}
