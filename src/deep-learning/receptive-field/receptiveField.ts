/**
 * Receptive fields: which input pixels can influence one unit deep inside a CNN.
 *
 * Every layer here is square and acts the same way along x and y, so the geometry is
 * worked out along one axis. For a layer with kernel k, stride s, padding p and
 * dilation d, going from layer l−1 to layer l:
 *
 *   size   n_l = ⌊(n_{l−1} + 2p − d(k−1) − 1) / s⌋ + 1        output width
 *   jump   j_l = j_{l−1} · s                                  input pixels between neighbouring units
 *   field  r_l = r_{l−1} + d(k−1) · j_{l−1}                   receptive field width
 *   start  c_l = c_{l−1} + (d(k−1)/2 − p) · j_{l−1}           input position of unit 0's centre
 *
 * (Pooling layers have the same geometry as convolutions; only what they compute differs.)
 */

export type LayerKind = 'conv' | 'pool';

export interface LayerSpec {
  kind: LayerKind;
  kernel: number;
  stride: number;
  padding: number;
  dilation: number;
}

export interface LayerGeometry {
  /** Units along one side. */
  size: number;
  /** Distance, in input pixels, between neighbouring units. */
  jump: number;
  /** Receptive field width in input pixels (it may extend past the image border). */
  field: number;
  /** Input coordinate of the centre of unit 0's receptive field. */
  start: number;
}

/** Geometry of the input (index 0) and of every layer's output (index i + 1). */
export function layerGeometry(inputSize: number, layers: LayerSpec[]): LayerGeometry[] {
  const geometry: LayerGeometry[] = [{ size: inputSize, jump: 1, field: 1, start: 0 }];
  for (const { kernel: k, stride: s, padding: p, dilation: d } of layers) {
    const prev = geometry[geometry.length - 1];
    const span = d * (k - 1);
    geometry.push({
      size: Math.floor((prev.size + 2 * p - span - 1) / s) + 1,
      jump: prev.jump * s,
      field: prev.field + span * prev.jump,
      start: prev.start + (span / 2 - p) * prev.jump,
    });
  }
  return geometry;
}

/** The input pixel range [lo, hi] (inclusive, possibly outside the image) seen by unit i of a layer. */
export function fieldRange(g: LayerGeometry, i: number): [number, number] {
  const centre = g.start + i * g.jump;
  return [centre - (g.field - 1) / 2, centre + (g.field - 1) / 2];
}

/**
 * Units [lo, hi] of layer `from` (0 = input) that unit range [lo, hi] of layer `to`
 * reads, clipped to the layer — the padding outside contributes nothing.
 */
export function dependencyRange(
  layers: LayerSpec[], geometry: LayerGeometry[], to: number, range: [number, number], from: number,
): [number, number] {
  let [lo, hi] = range;
  for (let l = to; l > from; l--) {
    const { kernel: k, stride: s, padding: p, dilation: d } = layers[l - 1];
    lo = lo * s - p;
    hi = hi * s - p + d * (k - 1);
    const n = geometry[l - 1].size;
    lo = Math.max(0, lo);
    hi = Math.min(n - 1, hi);
  }
  return [lo, hi];
}

/**
 * How many distinct paths connect each input pixel (along one axis) to unit `i` of
 * layer `to`. Centre pixels are reached by many overlapping windows, border pixels of
 * the field by only one: the "effective" receptive field is much more peaked than the
 * theoretical box. (In 2-D the count at (x, y) is countsX[x] · countsY[y].)
 */
export function pathCounts(layers: LayerSpec[], geometry: LayerGeometry[], to: number, i: number): Float64Array {
  let counts = new Float64Array(geometry[to].size);
  counts[i] = 1;
  for (let l = to; l > 0; l--) {
    const { kernel: k, stride: s, padding: p, dilation: d } = layers[l - 1];
    const prev = new Float64Array(geometry[l - 1].size);
    for (let u = 0; u < counts.length; u++) {
      if (counts[u] === 0) continue;
      for (let t = 0; t < k; t++) {
        const x = u * s - p + t * d;
        if (x >= 0 && x < prev.length) prev[x] += counts[u];
      }
    }
    counts = prev;
  }
  return counts;
}

export interface Preset {
  label: string;
  inputSize: number;
  layers: LayerSpec[];
}

const conv = (kernel: number, stride = 1, padding = 0, dilation = 1): LayerSpec => ({ kind: 'conv', kernel, stride, padding, dilation });
const pool = (kernel = 2, stride = kernel, padding = 0): LayerSpec => ({ kind: 'pool', kernel, stride, padding, dilation: 1 });

export const PRESETS: Record<string, Preset> = {
  lenet: { label: 'LeNet-5', inputSize: 28, layers: [conv(5), pool(), conv(5), pool()] },
  vgg: { label: 'VGG-style (3×3 + pooling)', inputSize: 32, layers: [conv(3, 1, 1), conv(3, 1, 1), pool(), conv(3, 1, 1), conv(3, 1, 1), pool()] },
  stacked: { label: 'Only 3×3 convolutions', inputSize: 32, layers: [conv(3, 1, 1), conv(3, 1, 1), conv(3, 1, 1), conv(3, 1, 1), conv(3, 1, 1)] },
  resnet: { label: 'ResNet stem (7×7 / 2, pool / 2)', inputSize: 64, layers: [conv(7, 2, 3), pool(3, 2, 1), conv(3, 1, 1), conv(3, 1, 1)] },
  dilated: { label: 'Dilated 3×3 (d = 1, 2, 4, 8)', inputSize: 48, layers: [conv(3, 1, 1, 1), conv(3, 1, 2, 2), conv(3, 1, 4, 4), conv(3, 1, 8, 8)] },
};
