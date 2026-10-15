/**
 * SLIC superpixels (Simple Linear Iterative Clustering, Achanta et al. 2012).
 *
 * k-means on 5-D points (L, a, b, x, y), with two twists that make it fast and give
 * compact, roughly equal-sized regions:
 *   1. Centres start on a regular grid with spacing S = √(N / K) (N pixels, K superpixels),
 *      nudged to the lowest-gradient pixel nearby so they don't sit on an edge.
 *   2. Each centre only competes for pixels within a 2S × 2S window around it, so one
 *      iteration costs O(N) instead of O(N·K).
 * The distance mixes colour and position:
 *     D = √(d_lab² + (d_xy / S)² · m²)
 * where the compactness m trades boundary adherence (small m) for regular shapes (large m).
 * A final pass merges fragments that ended up disconnected from their centre.
 */

export interface Lab {
  L: Float64Array;
  a: Float64Array;
  b: Float64Array;
}

/** sRGB (0–255) → CIELAB (D65). Lab distances roughly match perceived colour differences. */
export function rgbToLab(rgba: Uint8ClampedArray, n: number): Lab {
  const L = new Float64Array(n), A = new Float64Array(n), B = new Float64Array(n);
  const lin = (c: number) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  for (let i = 0; i < n; i++) {
    const r = lin(rgba[i * 4]), g = lin(rgba[i * 4 + 1]), b = lin(rgba[i * 4 + 2]);
    const x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    const fx = f(x), fy = f(y), fz = f(z);
    L[i] = 116 * fy - 16;
    A[i] = 500 * (fx - fy);
    B[i] = 200 * (fy - fz);
  }
  return { L, a: A, b: B };
}

export interface Centre {
  L: number;
  a: number;
  b: number;
  x: number;
  y: number;
}

export interface SlicState {
  centres: Centre[];
  labels: Int32Array;
  /** Grid spacing S. */
  step: number;
}

/** Grid of centres, each moved to the lowest-gradient pixel in its 3×3 neighbourhood. */
export function initCentres(lab: Lab, width: number, height: number, k: number): SlicState {
  const step = Math.max(2, Math.sqrt((width * height) / k));
  const { L, a, b } = lab;
  const at = (x: number, y: number) => Math.min(width - 1, Math.max(0, x)) + Math.min(height - 1, Math.max(0, y)) * width;
  const gradient = (x: number, y: number) => {
    const dx = at(x + 1, y), sx = at(x - 1, y), dy = at(x, y + 1), sy = at(x, y - 1);
    return (L[dx] - L[sx]) ** 2 + (a[dx] - a[sx]) ** 2 + (b[dx] - b[sx]) ** 2
      + (L[dy] - L[sy]) ** 2 + (a[dy] - a[sy]) ** 2 + (b[dy] - b[sy]) ** 2;
  };
  const centres: Centre[] = [];
  for (let gy = step / 2; gy < height; gy += step) {
    for (let gx = step / 2; gx < width; gx += step) {
      let bx = Math.floor(gx), by = Math.floor(gy), best = Infinity;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const x = Math.floor(gx) + dx, y = Math.floor(gy) + dy;
        if (x < 0 || y < 0 || x >= width || y >= height) continue;
        const g = gradient(x, y);
        if (g < best) { best = g; bx = x; by = y; }
      }
      const i = by * width + bx;
      centres.push({ L: L[i], a: a[i], b: b[i], x: bx, y: by });
    }
  }
  return { centres, labels: new Int32Array(width * height).fill(-1), step };
}

/** One SLIC iteration: assign pixels to the nearest centre in its 2S window, then move centres to their pixels' mean. */
export function slicIteration(lab: Lab, width: number, height: number, state: SlicState, compactness: number): SlicState {
  const { centres, step } = state;
  const { L, a, b } = lab;
  const labels = new Int32Array(width * height).fill(-1);
  const dist = new Float64Array(width * height).fill(Infinity);
  const spatial = (compactness / step) ** 2;
  centres.forEach((c, k) => {
    const x0 = Math.max(0, Math.floor(c.x - step)), x1 = Math.min(width - 1, Math.ceil(c.x + step));
    const y0 = Math.max(0, Math.floor(c.y - step)), y1 = Math.min(height - 1, Math.ceil(c.y + step));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const i = y * width + x;
      const d = (L[i] - c.L) ** 2 + (a[i] - c.a) ** 2 + (b[i] - c.b) ** 2 + ((x - c.x) ** 2 + (y - c.y) ** 2) * spatial;
      if (d < dist[i]) { dist[i] = d; labels[i] = k; }
    }
  });
  const sums = centres.map(() => [0, 0, 0, 0, 0, 0]);
  for (let i = 0; i < labels.length; i++) {
    const k = labels[i];
    if (k < 0) continue;
    const s = sums[k];
    s[0] += L[i]; s[1] += a[i]; s[2] += b[i]; s[3] += i % width; s[4] += Math.floor(i / width); s[5]++;
  }
  const moved = centres.map((c, k) => {
    const s = sums[k];
    return s[5] ? { L: s[0] / s[5], a: s[1] / s[5], b: s[2] / s[5], x: s[3] / s[5], y: s[4] / s[5] } : c;
  });
  return { centres: moved, labels, step };
}

/**
 * Give every pixel a label of a 4-connected region: each connected piece of a cluster
 * smaller than a quarter of the expected size is merged into the neighbouring region
 * it touches first. Returns labels numbered 0..count-1.
 */
export function enforceConnectivity(labels: Int32Array, width: number, height: number, step: number): { labels: Int32Array; count: number } {
  const n = width * height;
  const out = new Int32Array(n).fill(-1);
  const minSize = Math.max(1, Math.floor((step * step) / 4));
  const queue = new Int32Array(n);
  let count = 0;
  for (let start = 0; start < n; start++) {
    if (out[start] >= 0) continue;
    // Flood the piece of `labels[start]` that contains `start`.
    let head = 0, tail = 0, adjacent = -1;
    queue[tail++] = start;
    out[start] = count;
    while (head < tail) {
      const i = queue[head++], x = i % width, y = (i - x) / width;
      for (const j of [x > 0 ? i - 1 : -1, x < width - 1 ? i + 1 : -1, y > 0 ? i - width : -1, y < height - 1 ? i + width : -1]) {
        if (j < 0) continue;
        if (out[j] < 0 && labels[j] === labels[start]) { out[j] = count; queue[tail++] = j; }
        else if (out[j] >= 0 && out[j] !== count && adjacent < 0) adjacent = out[j];
      }
    }
    if (tail < minSize && adjacent >= 0) {
      for (let q = 0; q < tail; q++) out[queue[q]] = adjacent;
    } else count++;
  }
  return { labels: out, count };
}

export interface SlicResult {
  labels: Int32Array;
  count: number;
  centres: Centre[];
  step: number;
}

/** The whole algorithm: init, `iterations` rounds of assign/update, then connectivity. */
export function slic(rgba: Uint8ClampedArray, width: number, height: number, k: number, compactness: number, iterations: number): SlicResult {
  const lab = rgbToLab(rgba, width * height);
  let state = initCentres(lab, width, height, k);
  for (let it = 0; it < iterations; it++) state = slicIteration(lab, width, height, state, compactness);
  if (iterations === 0) return { labels: state.labels, count: 0, centres: state.centres, step: state.step };
  const { labels, count } = enforceConnectivity(state.labels, width, height, state.step);
  return { labels, count, centres: state.centres, step: state.step };
}

/** Each region painted with its mean colour. */
export function meanColours(rgba: Uint8ClampedArray, labels: Int32Array, count: number): Uint8ClampedArray<ArrayBuffer> {
  const sums = new Float64Array(count * 4);
  for (let i = 0; i < labels.length; i++) {
    const k = labels[i];
    if (k < 0) continue;
    sums[k * 4] += rgba[i * 4]; sums[k * 4 + 1] += rgba[i * 4 + 1]; sums[k * 4 + 2] += rgba[i * 4 + 2]; sums[k * 4 + 3]++;
  }
  const out = new Uint8ClampedArray(labels.length * 4);
  for (let i = 0; i < labels.length; i++) {
    const k = labels[i], c = sums[k * 4 + 3] || 1;
    for (let ch = 0; ch < 3; ch++) out[i * 4 + ch] = k < 0 ? rgba[i * 4 + ch] : sums[k * 4 + ch] / c;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** True where a pixel's right or lower neighbour has another label. */
export function boundaries(labels: Int32Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(labels.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = y * width + x;
    if ((x < width - 1 && labels[i + 1] !== labels[i]) || (y < height - 1 && labels[i + width] !== labels[i])) out[i] = 1;
  }
  return out;
}

/**
 * Boundary recall: the fraction of true edge pixels that have a superpixel boundary within
 * `tolerance` px. The standard way to score how well superpixels follow object outlines.
 */
export function boundaryRecall(truthEdges: Uint8Array, predicted: Uint8Array, width: number, height: number, tolerance = 1): number {
  let total = 0, hit = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (!truthEdges[y * width + x]) continue;
    total++;
    search: for (let dy = -tolerance; dy <= tolerance; dy++) for (let dx = -tolerance; dx <= tolerance; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < width && yy < height && predicted[yy * width + xx]) { hit++; break search; }
    }
  }
  return total ? hit / total : 1;
}
