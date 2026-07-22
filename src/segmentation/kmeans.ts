/**
 * K-means clustering for image segmentation.
 * Given grayscale pixel values (0–255), assigns each pixel to one of k clusters.
 */

export interface KMeansResult {
  /** Cluster centroids (mean intensity per cluster, 0–255) */
  centroids: Float64Array;
  /** Cluster assignment for each pixel (0 .. k-1) */
  labels: Uint8Array;
  /** Number of iterations performed */
  iterations: number;
}

/**
 * Run K-means clustering on grayscale pixel values.
 * @param pixels - Flat array of grayscale values (0–255)
 * @param k - Number of clusters (e.g. 2 or 3)
 * @param maxIter - Maximum iterations
 * @returns Centroid values, pixel labels, and iteration count
 */
export function kmeans(pixels: Float64Array, k: number, maxIter = 50): KMeansResult {
  const n = pixels.length;

  // Initialise centroids: evenly spaced across min–max value range
  const sorted = new Float64Array(pixels);
  sorted.sort((a, b) => a - b);
  const minVal = sorted[0];
  const maxVal = sorted[n - 1];
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

    // Assign each pixel to nearest centroid
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

    // Update centroids
    const sums = new Float64Array(k);
    const counts = new Uint32Array(k);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      sums[c] += pixels[i];
      counts[c]++;
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) centroids[c] = sums[c] / counts[c];
    }
  }

  return { centroids, labels, iterations };
}

/**
 * Convert cluster labels to a displayable image.
 * Each pixel gets the centroid value of its assigned cluster.
 */
export function labelsToImage(pixels: Float64Array, labels: Uint8Array, centroids: Float64Array): Float64Array {
  const out = new Float64Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    out[i] = centroids[labels[i]];
  }
  return out;
}

/**
 * Build an RGB-coloured segmentation map from labels.
 * Each cluster gets a distinct colour.
 */
export function labelsToColourImage(labels: Uint8Array, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const colours: number[][] = [
    [108, 140, 255],  // blue
    [248, 113, 113],  // red
    [52, 211, 153],   // green
    [251, 191, 36],   // yellow
    [196, 181, 253],  // purple
  ];
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const c = labels[i] % colours.length;
    rgba[i * 4] = colours[c][0];
    rgba[i * 4 + 1] = colours[c][1];
    rgba[i * 4 + 2] = colours[c][2];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
