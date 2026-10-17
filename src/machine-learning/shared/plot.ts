/** Plot coordinates and decision maps for ScatterPlot. */

/** The plotted range of each axis, in data units. */
export interface Extent {
  x: [number, number];
  y: [number, number];
}

export const UNIT_EXTENT: Extent = { x: [-1.5, 1.5], y: [-1.5, 1.5] };

/** Class c is drawn in CLASS_COLORS[c] (CSS colours, resolved for the current theme). */
export const CLASS_COLORS = ['var(--series-blue)', 'var(--series-orange)', 'var(--series-green)', 'var(--series-pink)', 'var(--series-red)', 'var(--series-light-green)'];

/** Class probabilities on a res × res grid over the extent: probs[(row · res + col) · classes + c], row 0 at the top. */
export interface ClassMap {
  res: number;
  classes: number;
  probs: Float32Array;
}

/**
 * Evaluate a classifier over the plot: `predict` returns either a class index or the
 * probability of every class at (x, y).
 */
export function classMap(predict: (x: number, y: number) => number | ArrayLike<number>, classes: number, res = 64, extent = UNIT_EXTENT): ClassMap {
  const probs = new Float32Array(res * res * classes);
  for (let row = 0; row < res; row++) for (let col = 0; col < res; col++) {
    const x = extent.x[0] + ((col + 0.5) / res) * (extent.x[1] - extent.x[0]);
    const y = extent.y[1] - ((row + 0.5) / res) * (extent.y[1] - extent.y[0]);
    const p = predict(x, y), at = (row * res + col) * classes;
    if (typeof p === 'number') probs[at + p] = 1;
    else for (let c = 0; c < classes; c++) probs[at + c] = p[c];
  }
  return { res, classes, probs };
}
