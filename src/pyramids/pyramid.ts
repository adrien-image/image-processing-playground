/**
 * Gaussian and Laplacian pyramids (Burt & Adelson, 1983) and multiband blending.
 *
 * Gaussian pyramid: G₀ = image, G₁ = reduce(G₀), G₂ = reduce(G₁), … — each level is a
 * blurred copy at half the resolution of the previous one.
 *
 * Laplacian pyramid: Lᵢ = Gᵢ − expand(Gᵢ₊₁) — the detail lost between two Gaussian
 * levels, i.e. one band of frequencies. The last level keeps the smallest Gaussian, so
 * the image can be rebuilt exactly by adding the bands back from coarse to fine.
 */
import { addPlanes, createPlane, gaussianBlur, subtractPlanes, type Plane } from './plane';

/**
 * The 5-tap binomial filter [1 4 6 4 1] / 16 — a cheap, smooth approximation of a Gaussian (σ ≈ 1).
 * It is a compromise, not an ideal low-pass: at the new Nyquist frequency (1/4 cycle per
 * pixel before halving) it still passes 25 %, so each level is *nearly* alias-free.
 */
export const BINOMIAL_5 = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16] as const;

const clampIndex = (i: number, n: number) => (i < 0 ? 0 : i >= n ? n - 1 : i);

// reduce and expand process one row at a time and write the result **transposed** (row y
// of the input becomes column y of the output). Doing it twice handles the rows, then the
// columns, and leaves the image the right way round — so each step needs only a row loop.

/** Blur every row with BINOMIAL_5 (replicate borders), keep the even pixels, write transposed. */
function reduceRowsTransposed(src: Float64Array, w: number, h: number): Float64Array {
  const nw = Math.ceil(w / 2);
  const out = new Float64Array(nw * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < nw; x++) {
      let sum = 0; // only the kept pixels (2x) are computed
      for (let k = -2; k <= 2; k++) sum += BINOMIAL_5[k + 2] * src[y * w + clampIndex(2 * x + k, w)];
      out[x * h + y] = sum;
    }
  }
  return out;
}

/**
 * One pyramid step down: blur with BINOMIAL_5 (rows then columns, replicate borders),
 * then keep every second pixel. The result is ⌈w/2⌉ × ⌈h/2⌉.
 *
 * The kept pixels are 0, 2, 4, …: coarse pixel i sits on fine pixel 2i, so after L steps
 * pixel i stands for the top-left pixel of its 2ᴸ × 2ᴸ block (the grid is not centred).
 */
export function reduce(plane: Plane): Plane {
  const { width: w, height: h } = plane;
  const nw = Math.ceil(w / 2), nh = Math.ceil(h / 2);
  const columnsAsRows = reduceRowsTransposed(plane.data, w, h); // h × nw, transposed
  return { data: reduceRowsTransposed(columnsAsRows, h, nw), width: nw, height: nh };
}

/**
 * Upsample every row to length `outW` and write transposed — the classic "insert a zero
 * between samples, then blur with 2 · [1 4 6 4 1]/16" (the 2 restores the brightness the
 * zeros removed). Written out, only two cases remain:
 *   even output 2i   = (s[i−1] + 6·s[i] + s[i+1]) / 8
 *   odd  output 2i+1 = (s[i] + s[i+1]) / 2
 * Computing it this way on the small grid lets replicate borders keep constants constant.
 */
function expandRowsTransposed(src: Float64Array, w: number, h: number, outW: number): Float64Array {
  const out = new Float64Array(outW * h);
  for (let y = 0; y < h; y++) {
    const s = (i: number) => src[y * w + clampIndex(i, w)];
    for (let o = 0; o < outW; o++) {
      const i = o >> 1;
      out[o * h + y] = o % 2 === 0 ? (s(i - 1) + 6 * s(i) + s(i + 1)) / 8 : (s(i) + s(i + 1)) / 2;
    }
  }
  return out;
}

/**
 * One pyramid step up: upsample to `width × height` (normally the size of the finer level,
 * at most twice the plane's size), rows then columns. Away from the borders this equals
 * zero-insertion followed by a 2-D blur with 4 · BINOMIAL_5 ⊗ BINOMIAL_5.
 */
export function expand(plane: Plane, width: number, height: number): Plane {
  const columnsAsRows = expandRowsTransposed(plane.data, plane.width, plane.height, width); // plane.height × width, transposed
  return { data: expandRowsTransposed(columnsAsRows, plane.height, width, height), width, height };
}

/** Number of levels until the smaller side reaches 1 pixel (256 × 256 → 9 levels: 256, 128, …, 1). */
export function maxLevels(width: number, height: number): number {
  let levels = 1;
  for (let side = Math.min(width, height); side > 1; side = Math.ceil(side / 2)) levels++;
  return levels;
}

/**
 * Deepest pyramid worth using for blending: count levels while the coarsest one still has
 * at least `minSide` pixels on its shorter side (256 × 256 → 5 levels, down to 16 × 16).
 *
 * Going deeper is possible but backfires. A 2 × 2 or 1 × 1 residual is little more than
 * each image's mean colour, and expand spreads it over the whole picture — so A's colours
 * leak far into B's side. And because reduce keeps pixels 0, 2, 4, …, a coarse sample can
 * sit right on the seam (mask ≈ 0.5 there), which makes the leak lopsided.
 */
export function blendLevelLimit(width: number, height: number, minSide = 16): number {
  let levels = 1;
  for (let side = Math.min(width, height); Math.ceil(side / 2) >= minSide; side = Math.ceil(side / 2)) levels++;
  return levels;
}

/** [G₀ = the plane itself, G₁, …, G_{levels−1}] with Gᵢ₊₁ = reduce(Gᵢ). */
export function gaussianPyramid(plane: Plane, levels: number): Plane[] {
  const pyramid = [plane];
  while (pyramid.length < levels) pyramid.push(reduce(pyramid[pyramid.length - 1]));
  return pyramid;
}

/**
 * Laplacian pyramid from a Gaussian one: Lᵢ = Gᵢ − expand(Gᵢ₊₁) (signed, mostly near 0:
 * only edges and texture of that scale survive), and the last level = the smallest
 * Gaussian level (the low-pass residual).
 */
export function laplacianFromGaussian(gaussian: readonly Plane[]): Plane[] {
  return gaussian.map((g, i) => (i === gaussian.length - 1 ? g : subtractPlanes(g, expand(gaussian[i + 1], g.width, g.height))));
}

/**
 * The `levels`-level Laplacian pyramid, read off deeper pyramids of the same image: the
 * bands L₀ … L_{levels−2} do not depend on the depth; only the residual (G_{levels−1}) does.
 * Handy for a "levels" slider: build the deepest pyramid once, then just slice.
 */
export function laplacianLevels(gaussian: readonly Plane[], laplacian: readonly Plane[], levels: number): Plane[] {
  return [...laplacian.slice(0, levels - 1), gaussian[levels - 1]];
}

/** [L₀, …, L_{levels−1}] of a plane (see `laplacianFromGaussian`). */
export function laplacianPyramid(plane: Plane, levels: number): Plane[] {
  return laplacianFromGaussian(gaussianPyramid(plane, levels));
}

/** Undo `laplacianPyramid`: start from the residual, then repeatedly expand and add the next band. */
export function reconstruct(laplacian: readonly Plane[]): Plane {
  let image = laplacian[laplacian.length - 1];
  for (let i = laplacian.length - 2; i >= 0; i--) {
    image = addPlanes(laplacian[i], expand(image, laplacian[i].width, laplacian[i].height));
  }
  return image;
}

/** Per pixel: alpha · a + (1 − alpha) · b, with alpha in 0–1. */
export function alphaBlend(a: Plane, b: Plane, alpha: Plane): Plane {
  const out = createPlane(a.width, a.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = alpha.data[i] * a.data[i] + (1 - alpha.data[i]) * b.data[i];
  return out;
}

/** Take a where the mask is ≥ 0.5, b elsewhere (a soft mask is thresholded): a visible seam wherever a and b differ. */
export function hardCut(a: Plane, b: Plane, mask: Plane): Plane {
  const out = createPlane(a.width, a.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = mask.data[i] >= 0.5 ? a.data[i] : b.data[i];
  return out;
}

/**
 * Feathering: alpha-blend with a Gaussian-blurred mask. One transition width for every
 * frequency: too narrow shows the seam, too wide gives "ghosts" (both images half-visible).
 */
export function feather(a: Plane, b: Plane, mask: Plane, sigma: number): Plane {
  return alphaBlend(a, b, gaussianBlur(mask, sigma));
}

/**
 * Combine two Laplacian pyramids level by level, weighting each band with the matching
 * level of the mask's Gaussian pyramid: blendedᵢ = Mᵢ · LAᵢ + (1 − Mᵢ) · LBᵢ.
 * Mᵢ is blurrier at coarser levels, so coarse bands mix over a wide area and fine bands
 * over a narrow one.
 */
export function blendLevels(lapA: readonly Plane[], lapB: readonly Plane[], maskPyramid: readonly Plane[]): Plane[] {
  return lapA.map((la, i) => alphaBlend(la, lapB[i], maskPyramid[i]));
}

/**
 * Multiband blending (Burt & Adelson's "multiresolution spline"): blend the Laplacian
 * pyramids of a and b with the Gaussian pyramid of the mask (1 = take a), then rebuild.
 * With one level it is a plain alpha blend with the mask (the hard cut, for a binary mask);
 * more levels widen the transition of the low frequencies (see `blendLevelLimit` for how far).
 */
export function blend(a: Plane, b: Plane, mask: Plane, levels: number): Plane {
  return reconstruct(blendLevels(laplacianPyramid(a, levels), laplacianPyramid(b, levels), gaussianPyramid(mask, levels)));
}
