/**
 * Separating touching objects: distance transform + markers + watershed.
 *
 * Two touching coins form one blob in a thresholded image, so connected components count
 * them as one. But the distance from each foreground pixel to the background peaks at each
 * coin's centre, with a "saddle" where they touch. Treat −distance as a landscape: one
 * basin per coin. Put a marker (a water source) in each basin and flood: water rises level
 * by level, each pixel joins the basin whose water reaches it first, and where two floods
 * meet we draw a dam — the watershed line that splits the blob.
 */

export interface Disc {
  x: number;
  y: number;
  r: number;
}

/**
 * Random discs in clusters: most new discs are placed against an existing one so that
 * they overlap by about `overlap` of the smaller radius (0 = just touching).
 */
export function randomDiscs(width: number, height: number, count: number, overlap: number, random: () => number): Disc[] {
  const discs: Disc[] = [];
  for (let attempt = 0; discs.length < count && attempt < 5000; attempt++) {
    const r = 9 + random() * 8;
    let x = r + random() * (width - 2 * r), y = r + random() * (height - 2 * r);
    if (discs.length && random() < 0.75) {                       // attach to an existing disc
      const d = discs[Math.floor(random() * discs.length)], angle = random() * Math.PI * 2;
      const gap = d.r + r - overlap * Math.min(d.r, r);
      x = d.x + gap * Math.cos(angle);
      y = d.y + gap * Math.sin(angle);
      if (x < r || y < r || x > width - r || y > height - r) continue;
    }
    // Never closer to another disc than the chosen overlap allows.
    if (discs.some(d => Math.hypot(d.x - x, d.y - y) < d.r + r - overlap * Math.min(d.r, r) - 0.5)) continue;
    discs.push({ x, y, r });
  }
  return discs;
}

export function rasteriseDiscs(discs: Disc[], width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (const d of discs) {
    for (let y = Math.max(0, Math.floor(d.y - d.r)); y <= Math.min(height - 1, Math.ceil(d.y + d.r)); y++)
      for (let x = Math.max(0, Math.floor(d.x - d.r)); x <= Math.min(width - 1, Math.ceil(d.x + d.r)); x++)
        if ((x - d.x) ** 2 + (y - d.y) ** 2 <= d.r * d.r) mask[y * width + x] = 1;
  }
  return mask;
}

/**
 * 1-D squared distance transform of f (Felzenszwalb & Huttenlocher): the lower envelope of
 * parabolas (q − p)² + f(p), found in one left-to-right pass.
 */
function edt1d(f: Float64Array, n: number): Float64Array {
  const d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  let k = 0;
  v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  const meet = (q: number, p: number) => (f[q] + q * q - (f[p] + p * p)) / (2 * q - 2 * p);
  for (let q = 1; q < n; q++) {
    let s = meet(q, v[k]);
    while (s <= z[k]) s = meet(q, v[--k]);                // the new parabola hides the last one: drop it
    k++;
    v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) ** 2 + f[v[k]];
  }
  return d;
}

/** Exact Euclidean distance from each foreground pixel to the nearest background pixel (0 on the background). */
export function distanceTransform(mask: Uint8Array, width: number, height: number): Float64Array {
  const BIG = 1e12;
  const sq = new Float64Array(width * height);
  const col = new Float64Array(height);
  for (let x = 0; x < width; x++) {                      // columns first...
    for (let y = 0; y < height; y++) col[y] = mask[y * width + x] ? BIG : 0;
    const d = edt1d(col, height);
    for (let y = 0; y < height; y++) sq[y * width + x] = d[y];
  }
  const row = new Float64Array(width);
  for (let y = 0; y < height; y++) {                     // ...then rows, on the column result
    for (let x = 0; x < width; x++) row[x] = sq[y * width + x];
    const d = edt1d(row, width);
    for (let x = 0; x < width; x++) sq[y * width + x] = d[x];
  }
  return sq.map(v => (v >= BIG ? 0 : Math.sqrt(v)));
}

/**
 * Markers: pixels that are the highest within `radius` px and at least `minDistance` from the
 * background, grouped into connected plateaus (one marker each). Returns labels 1..count, 0 elsewhere.
 */
export function findMarkers(dist: Float64Array, width: number, height: number, radius: number, minDistance: number): { markers: Int32Array; count: number } {
  const peak = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const v = dist[y * width + x];
    if (v < minDistance) continue;
    let isMax = true;
    for (let dy = -radius; dy <= radius && isMax; dy++) for (let dx = -radius; dx <= radius; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= width || yy >= height || dx * dx + dy * dy > radius * radius) continue;
      if (dist[yy * width + xx] > v + 1e-9) { isMax = false; break; }
    }
    if (isMax) peak[y * width + x] = 1;
  }
  const markers = new Int32Array(width * height);
  let count = 0;
  const stack: number[] = [];
  for (let i = 0; i < peak.length; i++) {
    if (!peak[i] || markers[i]) continue;
    markers[i] = ++count;
    stack.push(i);
    while (stack.length) {
      const j = stack.pop()!, x = j % width, y = (j - x) / width;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy, k = yy * width + xx;
        if (xx >= 0 && yy >= 0 && xx < width && yy < height && peak[k] && !markers[k]) { markers[k] = count; stack.push(k); }
      }
    }
  }
  return { markers, count };
}

/** Binary min-heap of pixel indices keyed by (priority, insertion order). */
class Queue {
  private keys: number[] = [];
  private items: number[] = [];
  private order: number[] = [];
  private counter = 0;
  get size() { return this.items.length; }
  push(item: number, key: number) {
    this.keys.push(key); this.items.push(item); this.order.push(this.counter++);
    let i = this.items.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p); i = p;
    }
  }
  pop(): number {
    const top = this.items[0];
    const last = this.items.length - 1;
    this.swap(0, last);
    this.keys.pop(); this.items.pop(); this.order.pop();
    let i = 0;
    for (;;) {
      const l = 2 * i + 1, r = l + 1;
      let m = i;
      if (l < this.items.length && this.less(l, m)) m = l;
      if (r < this.items.length && this.less(r, m)) m = r;
      if (m === i) break;
      this.swap(i, m); i = m;
    }
    return top;
  }
  private less(a: number, b: number) {
    return this.keys[a] < this.keys[b] || (this.keys[a] === this.keys[b] && this.order[a] < this.order[b]);
  }
  private swap(a: number, b: number) {
    [this.keys[a], this.keys[b]] = [this.keys[b], this.keys[a]];
    [this.items[a], this.items[b]] = [this.items[b], this.items[a]];
    [this.order[a], this.order[b]] = [this.order[b], this.order[a]];
  }
}

export interface WatershedResult {
  /** Basin label per pixel (0 = background or watershed line). */
  labels: Int32Array;
  /** True on the dams between two basins. */
  lines: Uint8Array;
  /** The landscape height at which each pixel was flooded (Infinity if never). */
  floodedAt: Float64Array;
}

/**
 * Marker-controlled watershed (Meyer's flooding). `surface` is the landscape (lower floods
 * first); only pixels where `mask` is set are flooded. A pixel reached by two different
 * basins becomes a watershed line.
 */
export function watershed(surface: Float64Array, markers: Int32Array, mask: Uint8Array, width: number, height: number): WatershedResult {
  const n = width * height;
  const labels = new Int32Array(markers);
  const lines = new Uint8Array(n);
  const floodedAt = new Float64Array(n).fill(Infinity);
  const queued = new Uint8Array(n);
  const queue = new Queue();
  const neighbours = (i: number) => {
    const x = i % width, y = (i - x) / width, out: number[] = [];
    if (x > 0) out.push(i - 1);
    if (x < width - 1) out.push(i + 1);
    if (y > 0) out.push(i - width);
    if (y < height - 1) out.push(i + width);
    return out;
  };
  for (let i = 0; i < n; i++) {
    if (!labels[i]) continue;
    floodedAt[i] = surface[i];
    for (const j of neighbours(i)) if (!labels[j] && mask[j] && !queued[j]) { queued[j] = 1; queue.push(j, surface[j]); }
  }
  while (queue.size) {
    const i = queue.pop();
    let label = 0, conflict = false;
    for (const j of neighbours(i)) {
      if (labels[j] > 0) {
        if (label === 0) label = labels[j];
        else if (labels[j] !== label) conflict = true;
      }
    }
    floodedAt[i] = surface[i];
    if (conflict) { lines[i] = 1; continue; }            // two floods meet: build a dam
    labels[i] = label;
    for (const j of neighbours(i)) if (!labels[j] && mask[j] && !queued[j]) { queued[j] = 1; queue.push(j, Math.max(surface[j], surface[i])); }
  }
  return { labels, lines, floodedAt };
}
