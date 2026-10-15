/** Glue for the Regions section: mask clean-up, picking regions, and choosing which to list. */

import { morphology } from '../morphology/morphology';
import type { Region } from './components';

/**
 * Morphological opening (erosion, then dilation) of a 0/1 mask with a size×size square.
 * Removes every foreground blob that cannot contain a full size×size square. `morphology`
 * ignores pixels outside the image, so blobs touching the image edge are judged on the
 * part of the square that lies inside the image.
 */
export function openMask(mask: Uint8Array, width: number, height: number, size: number): Uint8Array {
  const opened = morphology(Float64Array.from(mask), width, height, 'open', size);
  return Uint8Array.from(opened, v => (v > 0.5 ? 1 : 0));
}

/**
 * The region at image position (x, y) (in pixels, fractional allowed), or null.
 * A point on a region pixel hits that region. Otherwise the nearest region pixel within
 * `tolerance` pixels wins (distance from the point to the pixel's square), so a near miss
 * on a 1-px speck still selects it while clicks in empty space, such as a ring's hole,
 * select nothing. Points outside the image return null.
 */
export function regionAt(
  labels: Int32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  tolerance = 0,
): number | null {
  if (!(x >= 0 && y >= 0 && x < width && y < height)) return null;
  const px = Math.floor(x), py = Math.floor(y);
  const direct = labels[py * width + px];
  if (direct > 0) return direct;

  const reach = Math.ceil(tolerance);
  let best = 0;
  let bestDistance = Infinity;
  for (let ny = Math.max(0, py - reach); ny <= Math.min(height - 1, py + reach); ny++) {
    for (let nx = Math.max(0, px - reach); nx <= Math.min(width - 1, px + reach); nx++) {
      const l = labels[ny * width + nx];
      if (l === 0) continue;
      // Distance from the point to the closest point of the unit square [nx, nx+1) × [ny, ny+1).
      const d = Math.hypot(Math.max(nx - x, 0, x - nx - 1), Math.max(ny - y, 0, y - ny - 1));
      if (d <= tolerance && d < bestDistance) {
        best = l;
        bestDistance = d;
      }
    }
  }
  return best > 0 ? best : null;
}

/**
 * The index (y·width + x) of the region's pixel with the highest value in `values`, the
 * image that was thresholded; ties go to the first pixel in raster order. When the
 * threshold rises this pixel is the last one to turn background, so remembering a
 * selected region by it lets the selection follow the region while the threshold,
 * opening, connectivity and minimum area change.
 */
export function strongestPixel(labels: Int32Array, values: Float64Array, width: number, region: Region): number {
  const { x0, y0, x1, y1 } = region.bbox;
  let best = -1;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * width + x;
      if (labels[i] === region.label && (best < 0 || values[i] > values[best])) best = i;
    }
  }
  return best;
}

/** The indices (y·width + x) of all the region's pixels, in raster order. */
export function regionPixels(labels: Int32Array, width: number, region: Region): Int32Array {
  const { x0, y0, x1, y1 } = region.bbox;
  const pixels = new Int32Array(region.area);
  let n = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * width + x;
      if (labels[i] === region.label) pixels[n++] = i;
    }
  }
  return pixels.subarray(0, n);
}

/**
 * The label that now covers a remembered region: the label at its anchor (its strongest
 * pixel) while the anchor is still foreground, otherwise the label covering most of the
 * region's remembered pixels (ties: lower label), or null when none of them is left.
 * The fallback keeps a selection alive when an opening or a higher threshold removes
 * just the anchor, e.g. because it lay on a thin part of the region.
 */
export function followRegion(labels: Int32Array, anchor: number, pixels: Int32Array): number | null {
  if (labels[anchor] > 0) return labels[anchor];
  const votes = new Map<number, number>();
  for (const i of pixels) {
    const l = labels[i];
    if (l > 0) votes.set(l, (votes.get(l) ?? 0) + 1);
  }
  let best = 0, bestVotes = 0;
  for (const [l, v] of votes) {
    if (v > bestVotes || (v === bestVotes && l < best)) {
      best = l;
      bestVotes = v;
    }
  }
  return best > 0 ? best : null;
}

/**
 * At most `max` regions to list and outline: all of them when there are few enough,
 * otherwise the `max` largest (ties: lower label first). The result is in label order.
 * Keeps the table and the overlay fast and readable on images with thousands of specks.
 */
export function largestRegions(regions: Region[], max: number): Region[] {
  if (regions.length <= max) return regions;
  return [...regions]
    .sort((a, b) => b.area - a.area || a.label - b.label)
    .slice(0, max)
    .sort((a, b) => a.label - b.label);
}
