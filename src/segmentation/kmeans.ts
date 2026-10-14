/**
 * K-means clustering for image segmentation.
 * Given grayscale pixel values (0–255), assigns each pixel to one of k clusters.
 */

export interface KMeansResult {
  /** Cluster centroids (mean intensity per cluster, 0–255), in ascending order */
  centroids: Float64Array;
  /** Cluster assignment for each pixel (0 .. k-1) */
  labels: Uint8Array;
  /** Number of iterations performed */
  iterations: number;
}

/** Distinct colours for cluster maps, as [r, g, b] and CSS. */
export const CLUSTER_COLOURS: readonly (readonly [number, number, number])[] = [
  [108, 140, 255], // blue
  [248, 113, 113], // red
  [52, 211, 153],  // green
  [251, 191, 36],  // yellow
  [196, 181, 253], // purple
];
export const clusterCss = (i: number) => {
  const [r, g, b] = CLUSTER_COLOURS[i % CLUSTER_COLOURS.length];
  return `rgb(${r}, ${g}, ${b})`;
};

/**
 * Run K-means clustering on grayscale pixel values.
 * @param pixels - Flat array of grayscale values (0–255)
 * @param k - Number of clusters (e.g. 2 or 3)
 * @param maxIter - Maximum iterations
 * @returns Centroid values, pixel labels, and iteration count
 */
export function kmeans(pixels: Float64Array, k: number, maxIter = 50): KMeansResult {
  const n = pixels.length;

  // Initialise centroids evenly across the value range (deterministic, so the
  // demo gives the same answer every time).
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    if (pixels[i] < minVal) minVal = pixels[i];
    if (pixels[i] > maxVal) maxVal = pixels[i];
  }
  const centroids = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    centroids[i] = minVal + ((maxVal - minVal) * (i + 0.5)) / k;
  }

  const labels = new Uint8Array(n);
  let iterations = 0;
  let changed = true;

  while (changed && iterations < maxIter) {
    changed = false;
    iterations++;

    // 1. Assign each pixel to its nearest centroid
    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const dist = Math.abs(pixels[i] - centroids[c]);
        if (dist < bestDist) {
          bestDist = dist;
          bestCluster = c;
        }
      }
      if (labels[i] !== bestCluster) {
        labels[i] = bestCluster;
        changed = true;
      }
    }

    // 2. Move each centroid to the mean of its pixels (empty clusters stay put)
    const sums = new Float64Array(k);
    const counts = new Uint32Array(k);
    for (let i = 0; i < n; i++) {
      sums[labels[i]] += pixels[i];
      counts[labels[i]]++;
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        const next = sums[c] / counts[c];
        if (next !== centroids[c]) changed = true;
        centroids[c] = next;
      }
    }
  }

  return { centroids, labels, iterations };
}

/** Replace each pixel by its cluster's centroid value. */
export function labelsToImage(labels: Uint8Array, centroids: Float64Array): Float64Array {
  const out = new Float64Array(labels.length);
  for (let i = 0; i < labels.length; i++) out[i] = centroids[labels[i]];
  return out;
}

/** RGBA map giving each cluster a distinct colour. */
export function labelsToColourImage(labels: Uint8Array, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const [r, g, b] = CLUSTER_COLOURS[labels[i] % CLUSTER_COLOURS.length];
    rgba[i * 4] = r;
    rgba[i * 4 + 1] = g;
    rgba[i * 4 + 2] = b;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
