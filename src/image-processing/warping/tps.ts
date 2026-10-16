/**
 * Thin-plate splines: a smooth, non-rigid warp from a handful of control points.
 *
 * Each output coordinate is an affine part plus a sum of radial bumps centred on the
 * control points:
 *     f(p) = a₀ + a₁·x + a₂·y + Σᵢ wᵢ · U(‖p − cᵢ‖),   U(r) = r² · log r²
 * Among all functions that take the required values at the control points, this one has
 * the smallest bending energy ∬ f_xx² + 2·f_xy² + f_yy²: it is the shape a thin metal
 * plate takes when pinned at those points. The weights solve one linear system
 *     [K + λI  P] [w]   [v]          K_ij = U(‖cᵢ − cⱼ‖),  P = [1  x  y]
 *     [Pᵀ      0] [a] = [0]
 * once for the x targets and once for the y targets. λ = 0 interpolates exactly; a
 * larger λ trades exactness for smoothness, and λ → ∞ leaves the best affine map.
 *
 * To warp an image we fit the inverse map, from target to source positions, and every
 * output pixel looks up its colour at f(pixel) (inverse mapping leaves no holes).
 */

export interface Point { x: number; y: number }

export interface Tps {
  centres: Point[];
  wx: number[];
  wy: number[];
  /** Affine parts (a₀, a₁, a₂) for x and y. */
  ax: [number, number, number];
  ay: [number, number, number];
}

/** The thin-plate kernel r²·log r² (0 at r = 0), from the squared distance. */
export const tpsKernel = (r2: number) => (r2 === 0 ? 0 : r2 * Math.log(r2));

/** Solve A·x = b by Gaussian elimination with partial pivoting. */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const p = M[col][col];
    if (Math.abs(p) < 1e-14) throw new Error('Singular system: control points must not all lie on a line');
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / p;
      if (f !== 0) for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return x;
}

/** The thin-plate spline mapping `src[i]` to `dst[i]` (≥ 3 points, not all collinear). */
export function fitTps(src: Point[], dst: Point[], lambda = 0): Tps {
  const n = src.length;
  const A: number[][] = Array.from({ length: n + 3 }, () => new Array<number>(n + 3).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) A[i][j] = tpsKernel((src[i].x - src[j].x) ** 2 + (src[i].y - src[j].y) ** 2);
    A[i][i] += lambda;
    const P = [1, src[i].x, src[i].y];
    for (let k = 0; k < 3; k++) { A[i][n + k] = P[k]; A[n + k][i] = P[k]; }
  }
  const zeros = [0, 0, 0];
  const solX = solveLinear(A, [...dst.map(p => p.x), ...zeros]);
  const solY = solveLinear(A, [...dst.map(p => p.y), ...zeros]);
  return {
    centres: src.map(p => ({ ...p })),
    wx: solX.slice(0, n), wy: solY.slice(0, n),
    ax: solX.slice(n) as [number, number, number],
    ay: solY.slice(n) as [number, number, number],
  };
}

export function applyTps(t: Tps, p: Point): Point {
  let x = t.ax[0] + t.ax[1] * p.x + t.ax[2] * p.y;
  let y = t.ay[0] + t.ay[1] * p.x + t.ay[2] * p.y;
  t.centres.forEach((c, i) => {
    const u = tpsKernel((p.x - c.x) ** 2 + (p.y - c.y) ** 2);
    x += t.wx[i] * u;
    y += t.wy[i] * u;
  });
  return { x, y };
}

/** Bending energy (up to a constant factor 8π): wₓᵀKwₓ + w_yᵀKw_y. Zero for an affine map. */
export function bendingEnergy(t: Tps): number {
  let e = 0;
  t.centres.forEach((a, i) => t.centres.forEach((b, j) => {
    const k = tpsKernel((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    e += k * (t.wx[i] * t.wx[j] + t.wy[i] * t.wy[j]);
  }));
  return e;
}

/**
 * Warp an RGBA image: output pixel (x, y), in coordinates scaled to [0, 1] by `size`,
 * takes the colour at `inverse((x, y))`, sampled bilinearly (outside is black).
 */
export function warpImage(rgba: Uint8ClampedArray, w: number, h: number, inverse: Tps): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(w * h * 4);
  const size = Math.max(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const s = applyTps(inverse, { x: (x + 0.5) / size, y: (y + 0.5) / size });
      const sx = s.x * size - 0.5, sy = s.y * size - 0.5;
      const o = (y * w + x) * 4;
      out[o + 3] = 255;
      if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) continue;
      const x0 = Math.min(w - 2, Math.floor(sx)), y0 = Math.min(h - 2, Math.floor(sy));
      const fx = sx - x0, fy = sy - y0;
      for (let c = 0; c < 3; c++) {
        const i = (y0 * w + x0) * 4 + c;
        const top = rgba[i] * (1 - fx) + rgba[i + 4] * fx;
        const bottom = rgba[i + w * 4] * (1 - fx) + rgba[i + w * 4 + 4] * fx;
        out[o + c] = top * (1 - fy) + bottom * fy;
      }
    }
  }
  return out;
}
