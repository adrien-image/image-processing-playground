/**
 * Active contours ("snakes", Kass, Witkin & Terzopoulos 1988).
 *
 * A closed curve of n points moves to minimise
 *     E = Σ α·|xᵢ₊₁ − xᵢ|²  +  β·|xᵢ₊₁ − 2xᵢ + xᵢ₋₁|²  −  w·|∇(G_σ ⋆ I)|²(xᵢ)
 * The first term keeps it short (elasticity), the second keeps it smooth (rigidity), the
 * third pulls it onto strong edges. The internal terms are linear in the points: their
 * gradient is A·x, with A the circulant matrix of the stencil (β, −α−4β, 2α+6β, −α−4β, β).
 * A semi-implicit step treats them implicitly and the image force explicitly:
 *     (I + τ·A) · x_{t+1} = x_t + τ · F(x_t)
 * so (I + τA)⁻¹ is computed once and every step is a matrix product, stable even for
 * large τ. The edge force only reaches a few σ from an edge, so a "balloon" force along
 * the normal (Cohen 1991) inflates or deflates the snake until it meets one.
 */

export interface Point { x: number; y: number }

/** ∇(|∇(G_σ ⋆ I)|²), normalised so the strongest edge has magnitude 1: points towards edges. */
export interface ForceField {
  width: number;
  height: number;
  /** Edge strength |∇(G_σ ⋆ I)|², scaled to [0, 1]. */
  edges: Float64Array;
  fx: Float64Array;
  fy: Float64Array;
}

function blur(pixels: Float64Array, w: number, h: number, sigma: number): Float64Array {
  if (sigma <= 0) return pixels;
  const r = Math.ceil(3 * sigma);
  const k = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)));
  const sum = k.reduce((a, b) => a + b, 0);
  const pass = (src: Float64Array, dx: number, dy: number) => {
    const out = new Float64Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let i = -r; i <= r; i++) {
          const xx = Math.min(w - 1, Math.max(0, x + i * dx)), yy = Math.min(h - 1, Math.max(0, y + i * dy));
          s += k[i + r] * src[yy * w + xx];
        }
        out[y * w + x] = s / sum;
      }
    }
    return out;
  };
  return pass(pass(pixels, 1, 0), 0, 1);
}

/** Central differences with replicate borders. */
function gradient(p: Float64Array, w: number, h: number) {
  const gx = new Float64Array(w * h), gy = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      gx[y * w + x] = (p[y * w + Math.min(w - 1, x + 1)] - p[y * w + Math.max(0, x - 1)]) / 2;
      gy[y * w + x] = (p[Math.min(h - 1, y + 1) * w + x] - p[Math.max(0, y - 1) * w + x]) / 2;
    }
  }
  return { gx, gy };
}

export function forceField(gray: Float64Array, width: number, height: number, sigma: number): ForceField {
  const smooth = blur(gray, width, height, sigma);
  const { gx, gy } = gradient(smooth, width, height);
  const edges = gx.map((v, i) => v * v + gy[i] * gy[i]);
  let max = 0;
  for (const v of edges) max = Math.max(max, v);
  if (max > 0) for (let i = 0; i < edges.length; i++) edges[i] /= max;
  const f = gradient(edges, width, height);
  let fmax = 0;
  for (let i = 0; i < edges.length; i++) fmax = Math.max(fmax, Math.hypot(f.gx[i], f.gy[i]));
  if (fmax > 0) for (let i = 0; i < edges.length; i++) { f.gx[i] /= fmax; f.gy[i] /= fmax; }
  return { width, height, edges, fx: f.gx, fy: f.gy };
}

/** Bilinear sample of a field at a sub-pixel position (clamped to the image). */
function sampleField(field: Float64Array, w: number, h: number, x: number, y: number): number {
  const cx = Math.min(w - 1.001, Math.max(0, x)), cy = Math.min(h - 1.001, Math.max(0, y));
  const x0 = Math.floor(cx), y0 = Math.floor(cy), fx = cx - x0, fy = cy - y0;
  const i = y0 * w + x0;
  return (field[i] * (1 - fx) + field[i + 1] * fx) * (1 - fy) + (field[i + w] * (1 - fx) + field[i + w + 1] * fx) * fy;
}

/** (I + τA)⁻¹ for a closed snake of n points, row-major n×n (Gauss–Jordan elimination). */
export function snakeMatrix(n: number, alpha: number, beta: number, tau: number): Float64Array {
  const stencil = [beta, -alpha - 4 * beta, 2 * alpha + 6 * beta, -alpha - 4 * beta, beta];
  const M = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(2 * n);
    for (let k = -2; k <= 2; k++) row[(i + k + n) % n] += tau * stencil[k + 2];
    row[i] += 1;
    row[n + i] = 1;
    return row;
  });
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const p = M[col][col];
    for (let c = 0; c < 2 * n; c++) M[col][c] /= p;
    for (let r = 0; r < n; r++) {
      if (r === col || M[r][col] === 0) continue;
      const f = M[r][col];
      for (let c = 0; c < 2 * n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const inv = new Float64Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) inv[r * n + c] = M[r][n + c];
  return inv;
}

/** Twice the signed area: positive when the points run clockwise on screen (y down). */
function signedArea(points: Point[]): number {
  let a = 0;
  points.forEach((p, i) => { const q = points[(i + 1) % points.length]; a += p.x * q.y - q.x * p.y; });
  return a;
}

export interface SnakeForces {
  tau: number;
  /** Weight w of the edge force. */
  edgeWeight: number;
  /** Balloon force along the outward normal: > 0 inflates, < 0 deflates. */
  balloon: number;
}

/** One semi-implicit step: x ← (I + τA)⁻¹ (x + τ·F(x)). */
export function snakeStep(points: Point[], inverse: Float64Array, field: ForceField, f: SnakeForces): Point[] {
  const n = points.length;
  const orientation = Math.sign(signedArea(points)) || 1;
  const bx = new Float64Array(n), by = new Float64Array(n);
  points.forEach((p, i) => {
    const prev = points[(i - 1 + n) % n], next = points[(i + 1) % n];
    // Outward normal: the tangent rotated by 90°, flipped for the curve's orientation.
    const tx = next.x - prev.x, ty = next.y - prev.y, len = Math.hypot(tx, ty) || 1;
    const nx = (orientation * ty) / len, ny = (-orientation * tx) / len;
    const ex = sampleField(field.fx, field.width, field.height, p.x, p.y);
    const ey = sampleField(field.fy, field.width, field.height, p.x, p.y);
    bx[i] = p.x + f.tau * (f.edgeWeight * ex + f.balloon * nx);
    by[i] = p.y + f.tau * (f.edgeWeight * ey + f.balloon * ny);
  });
  return Array.from({ length: n }, (_, r) => {
    let x = 0, y = 0;
    for (let c = 0; c < n; c++) { x += inverse[r * n + c] * bx[c]; y += inverse[r * n + c] * by[c]; }
    return { x: Math.min(field.width - 1, Math.max(0, x)), y: Math.min(field.height - 1, Math.max(0, y)) };
  });
}

/** Redistribute the points evenly along the closed curve (stops them bunching up). */
export function resample(points: Point[], n = points.length): Point[] {
  const seg = points.map((p, i) => Math.hypot(points[(i + 1) % points.length].x - p.x, points[(i + 1) % points.length].y - p.y));
  const total = seg.reduce((a, b) => a + b, 0);
  if (total === 0) return points;
  const out: Point[] = [];
  let i = 0, walked = 0;
  for (let k = 0; k < n; k++) {
    const target = (k / n) * total;
    while (walked + seg[i] < target && i < points.length - 1) walked += seg[i++];
    const t = seg[i] ? (target - walked) / seg[i] : 0;
    const a = points[i], b = points[(i + 1) % points.length];
    out.push({ x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) });
  }
  return out;
}

export function circle(cx: number, cy: number, r: number, n: number): Point[] {
  return Array.from({ length: n }, (_, i) => ({ x: cx + r * Math.cos((2 * Math.PI * i) / n), y: cy + r * Math.sin((2 * Math.PI * i) / n) }));
}

export const perimeter = (points: Point[]) =>
  points.reduce((s, p, i) => s + Math.hypot(points[(i + 1) % points.length].x - p.x, points[(i + 1) % points.length].y - p.y), 0);
