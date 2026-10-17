/**
 * Support vector machines: the widest-margin classifier, trained with SMO.
 *
 * A linear SVM looks for the line (hyperplane) w·x + b = 0 that separates the two classes
 * y = ±1 with the widest empty band around it. Scaling w so that the closest points have
 * y·f(x) = 1, the band (the margin) is 2/‖w‖ wide, so the widest margin is the smallest ‖w‖.
 * The soft-margin version lets points cross into the band, or even the wrong side, at a
 * price C per unit of "hinge loss":
 *   minimise  ½‖w‖² + C · Σᵢ max(0, 1 − yᵢ f(xᵢ)),   f(x) = w·x + b
 * Small C buys a wide margin with many violations; large C insists on classifying every
 * training point, even at the cost of a narrow margin.
 *
 * The dual problem has one coefficient αᵢ per training point:
 *   maximise  Σ αᵢ − ½ Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)   subject to  0 ≤ αᵢ ≤ C,  Σ αᵢ yᵢ = 0
 *   f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b              (linear kernel: w = Σ αᵢ yᵢ xᵢ)
 * At the optimum most αᵢ are exactly 0: only the points on or inside the margin, the
 * support vectors, appear in f. The data enter only through the kernel K(xᵢ, xⱼ), a
 * similarity, so replacing x·z by another similarity (RBF, polynomial) gives a non-linear
 * boundary without ever computing the features: the kernel trick.
 *
 * SMO (Platt 1998) solves the dual two coefficients at a time: changing a single αᵢ would
 * break Σ αᵢ yᵢ = 0, but a pair can move along that constraint, and the best move for a
 * pair has a closed form (then clipped to the box [0, C]). The pair chosen is the one that
 * violates the optimality (KKT) conditions most, the heuristic LIBSVM uses, and training
 * stops when no pair violates them by more than `tol`.
 */

export type Kernel =
  | { type: 'linear' }
  /** exp(−γ‖x − z‖²): a bump of width ~1/√γ around every training point. */
  | { type: 'rbf'; gamma: number }
  /** (x·z + 1)^degree: every product of up to `degree` coordinates. */
  | { type: 'poly'; degree: number };

/** K(a, b) for rows `i` of `a` and `j` of `b` (both n × dim, row-major). */
export function kernelValue(kernel: Kernel, a: ArrayLike<number>, i: number, b: ArrayLike<number>, j: number, dim: number): number {
  if (kernel.type === 'rbf') {
    let d2 = 0;
    for (let c = 0; c < dim; c++) { const t = a[i * dim + c] - b[j * dim + c]; d2 += t * t; }
    return Math.exp(-kernel.gamma * d2);
  }
  let dot = 0;
  for (let c = 0; c < dim; c++) dot += a[i * dim + c] * b[j * dim + c];
  return kernel.type === 'linear' ? dot : (dot + 1) ** kernel.degree;
}

export interface SvmModel {
  kernel: Kernel;
  dim: number;
  /** The training points (n × dim) and labels ±1. */
  x: Float64Array;
  y: Int8Array;
  /** Dual coefficients, 0 ≤ αᵢ ≤ C. */
  alpha: Float64Array;
  b: number;
  C: number;
  /** SMO pair updates done, and whether the KKT gap fell below the tolerance. */
  iterations: number;
  converged: boolean;
}

export interface SmoOptions {
  /** Stop when the largest KKT violation is below this (LIBSVM's default is 1e-3). */
  tol?: number;
  maxIterations?: number;
}

/**
 * Train a soft-margin SVM on n points `x` (n × dim) with labels `y` ∈ {−1, +1}.
 */
export function trainSvm(x: ArrayLike<number>, y: ArrayLike<number>, dim: number, kernel: Kernel, C: number, { tol = 1e-3, maxIterations = 50_000 }: SmoOptions = {}): SvmModel {
  const n = y.length;
  const X = Float64Array.from(x);
  const Y = Int8Array.from(y, v => (v > 0 ? 1 : -1));
  // The kernel matrix, computed once: SMO looks entries up all the time.
  const K = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) K[i * n + j] = K[j * n + i] = kernelValue(kernel, X, i, X, j, dim);

  const alpha = new Float64Array(n);
  // u[t] = Σⱼ αⱼ yⱼ K(xⱼ, xₜ), the decision function without b. Starts at 0 (all α = 0).
  const u = new Float64Array(n);
  let iterations = 0, converged = false, bMin = 0, bMax = 0;

  while (iterations < maxIterations) {
    // KKT at the optimum, for some b: yₜ − uₜ ≤ b for every t whose αₜ could still move
    // so as to raise yₜf(xₜ) ("up"), and ≥ b for every t that could lower it ("low").
    // The worst violation is between the largest yₜ − uₜ among "up" and the smallest among "low".
    const up = (t: number) => (Y[t] > 0 ? alpha[t] < C : alpha[t] > 0);
    const low = (t: number) => (Y[t] > 0 ? alpha[t] > 0 : alpha[t] < C);
    let i = -1, gi = -Infinity, gMin = Infinity;
    for (let t = 0; t < n; t++) {
      const g = Y[t] - u[t];
      if (up(t) && g > gi) { gi = g; i = t; }
      if (low(t) && g < gMin) gMin = g;
    }
    bMin = gi; bMax = gMin; // at the optimum any b in [bMin, bMax] satisfies KKT
    if (i < 0 || gi - gMin < tol) { converged = true; break; }
    // Partner j: of the "low" points that violate KKT with i, the one whose pair step gains
    // the most dual objective, (gᵢ − gⱼ)² / ηᵢⱼ (Fan, Chen & Lin 2005, as in LIBSVM). Taking
    // simply the smallest gⱼ also converges, but far more slowly when C is large.
    let j = -1, best = -Infinity;
    for (let t = 0; t < n; t++) {
      const g = Y[t] - u[t];
      if (!low(t) || g >= gi) continue;
      const gain = (gi - g) ** 2 / Math.max(K[i * n + i] + K[t * n + t] - 2 * K[i * n + t], 1e-12);
      if (gain > best) { best = gain; j = t; }
    }

    // Move αᵢ and αⱼ along yᵢαᵢ + yⱼαⱼ = constant. With Eₜ = uₜ − yₜ (the error without b),
    // the dual objective along that line is a parabola of curvature η, maximal at:
    const Ei = u[i] - Y[i], Ej = u[j] - Y[j];
    const eta = Math.max(K[i * n + i] + K[j * n + j] - 2 * K[i * n + j], 1e-12);
    // The box 0 ≤ α ≤ C for both, expressed as bounds [L, H] on the new αⱼ.
    const L = Y[i] !== Y[j] ? Math.max(0, alpha[j] - alpha[i]) : Math.max(0, alpha[i] + alpha[j] - C);
    const H = Y[i] !== Y[j] ? Math.min(C, C + alpha[j] - alpha[i]) : Math.min(C, alpha[i] + alpha[j]);
    const ajNew = Math.min(H, Math.max(L, alpha[j] + (Y[j] * (Ei - Ej)) / eta));
    // αᵢ moves by the opposite amount (times yᵢyⱼ); round-off is snapped back into the box.
    const aiNew = snap(alpha[i] - Y[i] * Y[j] * (ajNew - alpha[j]), C);
    const di = aiNew - alpha[i], dj = ajNew - alpha[j];
    alpha[i] = aiNew;
    alpha[j] = ajNew;
    for (let t = 0; t < n; t++) u[t] += di * Y[i] * K[i * n + t] + dj * Y[j] * K[j * n + t];
    iterations++;
  }

  // b: every "free" support vector (0 < α < C) lies exactly on its margin, yf = 1, so
  // b = y − u there; average them for accuracy. Without any, b lies between the bounds.
  let sum = 0, free = 0;
  for (let t = 0; t < n; t++) if (alpha[t] > 1e-8 * C && alpha[t] < C * (1 - 1e-8)) { sum += Y[t] - u[t]; free++; }
  const b = free ? sum / free : (bMin + bMax) / 2;
  return { kernel, dim, x: X, y: Y, alpha, b, C, iterations, converged };
}

/** Clamp to [0, C], treating values within round-off of a bound as on it. */
function snap(a: number, C: number): number {
  return a < 1e-12 * C ? 0 : a > C * (1 - 1e-12) ? C : a;
}

/** f(z) = Σ αᵢ yᵢ K(xᵢ, z) + b for one point z (dim numbers); its sign is the class. */
export function decision(model: SvmModel, z: ArrayLike<number>): number {
  let f = model.b;
  for (let i = 0; i < model.y.length; i++) {
    if (model.alpha[i] === 0) continue; // only the support vectors count
    f += model.alpha[i] * model.y[i] * kernelValue(model.kernel, model.x, i, z, 0, model.dim);
  }
  return f;
}

/** Which points are support vectors (α > 0), and of those which sit at the bound C. */
export function supportVectors(model: SvmModel): { support: boolean[]; bounded: boolean[] } {
  const support = Array.from(model.alpha, a => a > 1e-8 * model.C);
  const bounded = Array.from(model.alpha, a => a >= model.C * (1 - 1e-8));
  return { support, bounded };
}

/** Linear kernel only: the primal weights w = Σ αᵢ yᵢ xᵢ. */
export function primalWeights(model: SvmModel): Float64Array {
  const w = new Float64Array(model.dim);
  for (let i = 0; i < model.y.length; i++) for (let c = 0; c < model.dim; c++) w[c] += model.alpha[i] * model.y[i] * model.x[i * model.dim + c];
  return w;
}

/**
 * The margin width 2/‖w‖, measured in the kernel's feature space, where
 * ‖w‖² = Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ). For the linear kernel it is the width of the band in the plane.
 */
export function marginWidth(model: SvmModel): number {
  let w2 = 0;
  const n = model.y.length;
  for (let i = 0; i < n; i++) {
    if (!model.alpha[i]) continue;
    for (let j = 0; j < n; j++) {
      if (!model.alpha[j]) continue;
      w2 += model.alpha[i] * model.alpha[j] * model.y[i] * model.y[j] * kernelValue(model.kernel, model.x, i, model.x, j, model.dim);
    }
  }
  return 2 / Math.sqrt(w2);
}

/** The primal objective ½‖w‖² + C Σ hinge, and how many points are inside their margin (yf < 1). */
export function objective(model: SvmModel): { value: number; hinge: number; violations: number } {
  const w = 2 / marginWidth(model);
  let hinge = 0, violations = 0;
  for (let i = 0; i < model.y.length; i++) {
    const m = model.y[i] * decision(model, model.x.subarray(i * model.dim, (i + 1) * model.dim));
    if (m < 1 - 1e-6) violations++;
    hinge += Math.max(0, 1 - m);
  }
  return { value: 0.5 * w * w + model.C * hinge, hinge, violations };
}
