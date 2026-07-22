/**
 * 2D FFT implementation for image frequency analysis.
 * Radix-2 Cooley-Tukey FFT. Works on images whose dimensions are powers of 2.
 */

// Complex number as [real, imag]
type Complex = [number, number];

/** 1D FFT (in-place, radix-2) */
function fft1d(data: Complex[]): void {
  const n = data.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) { const t = data[i]; data[i] = data[j]; data[j] = t; }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }

  // Cooley-Tukey iterations
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -2 * Math.PI / len;
    const wlen: Complex = [Math.cos(angle), Math.sin(angle)];
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let j = 0; j < len / 2; j++) {
        const u = data[i + j];
        const v: Complex = [data[i + j + len / 2][0] * wr - data[i + j + len / 2][1] * wi,
                           data[i + j + len / 2][1] * wr + data[i + j + len / 2][0] * wi];
        data[i + j] = [u[0] + v[0], u[1] + v[1]];
        data[i + j + len / 2] = [u[0] - v[0], u[1] - v[1]];
        const nw = wr * wlen[0] - wi * wlen[1];
        wi = wr * wlen[1] + wi * wlen[0];
        wr = nw;
      }
    }
  }
}

/** 1D IFFT (in-place) */
function ifft1d(data: Complex[]): void {
  // Conjugate
  for (let i = 0; i < data.length; i++) data[i][1] = -data[i][1];
  fft1d(data);
  // Conjugate and scale
  const n = data.length;
  for (let i = 0; i < n; i++) {
    data[i][1] = -data[i][1];
    data[i][0] /= n;
    data[i][1] /= n;
  }
}

/** 2D FFT: applies 1D FFT to each row, then each column */
export function fft2d(image: Complex[][]): Complex[][] {
  const h = image.length, w = image[0].length;
  const result = image.map(row => row.map(c => [c[0], c[1]] as Complex));

  // Rows
  for (let y = 0; y < h; y++) fft1d(result[y]);
  // Columns
  for (let x = 0; x < w; x++) {
    const col: Complex[] = [];
    for (let y = 0; y < h; y++) col.push(result[y][x]);
    fft1d(col);
    for (let y = 0; y < h; y++) result[y][x] = col[y];
  }
  return result;
}

/** 2D IFFT */
export function ifft2d(freq: Complex[][]): Complex[][] {
  const h = freq.length, w = freq[0].length;
  const result = freq.map(row => row.map(c => [c[0], c[1]] as Complex));

  // Rows
  for (let y = 0; y < h; y++) ifft1d(result[y]);
  // Columns
  for (let x = 0; x < w; x++) {
    const col: Complex[] = [];
    for (let y = 0; y < h; y++) col.push(result[y][x]);
    ifft1d(col);
    for (let y = 0; y < h; y++) result[y][x] = col[y];
  }
  return result;
}

/** Convert grayscale pixels to complex array (2D) */
export function pixelsToComplex(pixels: Float64Array, w: number, h: number): Complex[][] {
  const result: Complex[][] = [];
  for (let y = 0; y < h; y++) {
    const row: Complex[] = [];
    for (let x = 0; x < w; x++) row.push([pixels[y * w + x], 0]);
    result.push(row);
  }
  return result;
}

/** Compute log magnitude spectrum for display (shifted so DC is at centre) */
export function magnitudeSpectrum(freq: Complex[][], w: number, h: number): Float64Array {
  const mag = new Float64Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      mag[y * w + x] = Math.log(1 + Math.sqrt(freq[y][x][0] ** 2 + freq[y][x][1] ** 2));

  // Normalise to 0-255
  let max = 0;
  for (let i = 0; i < w * h; i++) if (mag[i] > max) max = mag[i];
  const s = max > 0 ? 255 / max : 1;
  for (let i = 0; i < w * h; i++) mag[i] *= s;

  // Shift: swap quadrants so DC is at centre
  const shifted = new Float64Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      shifted[y * w + x] = mag[((y + h / 2) % h) * w + ((x + w / 2) % w)];
  return shifted;
}

export type FilterKind = 'ideal' | 'gaussian';

/** Apply circular low-pass, high-pass, band-pass or band-stop filter in frequency domain */
export function applyFilter(freq: Complex[][], cutoff: number, type: 'low' | 'high' | 'band-pass' | 'band-stop', kind: FilterKind = 'ideal', cutoff2?: number): Complex[][] {
  const h = freq.length, w = freq[0].length;
  const cx = w / 2, cy = h / 2;
  const c2 = cutoff2 ?? cutoff;
  const result = freq.map(row => row.map(c => [c[0], c[1]] as Complex));

  const shifted: Complex[][] = [];
  for (let y = 0; y < h; y++) {
    shifted[y] = [];
    for (let x = 0; x < w; x++)
      shifted[y][x] = result[(y + h / 2) % h][(x + w / 2) % w];
  }

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      let gain: number;
      if (kind === 'ideal') {
        if (type === 'low') gain = dist <= cutoff ? 1 : 0;
        else if (type === 'high') gain = dist >= cutoff ? 1 : 0;
        else if (type === 'band-pass') gain = (dist >= cutoff && dist <= c2) ? 1 : 0;
        else gain = (dist >= cutoff && dist <= c2) ? 0 : 1; // band-stop
      } else {
        const sigma = cutoff / 2;
        gain = Math.exp(-(dist * dist) / (2 * sigma * sigma));
        if (type === 'high') gain = 1 - gain;
        else if (type === 'band-pass') {
          const sigma2 = c2 / 2;
          gain = Math.exp(-(dist * dist) / (2 * sigma2 * sigma2)) - gain;
          gain = Math.max(0, gain);
        } else if (type === 'band-stop') {
          const sigma2 = c2 / 2;
          gain = 1 - Math.exp(-(dist * dist) / (2 * sigma2 * sigma2)) + Math.exp(-(dist * dist) / (2 * sigma * sigma));
          gain = Math.min(1, gain);
        }
      }
      shifted[y][x][0] *= gain;
      shifted[y][x][1] *= gain;
    }
  }

  // Shift back
  const unshifted: Complex[][] = [];
  for (let y = 0; y < h; y++) {
    unshifted[y] = [];
    for (let x = 0; x < w; x++)
      unshifted[y][x] = shifted[(y + h / 2) % h][(x + w / 2) % w];
  }
  return unshifted;
}

/** Validate that dimensions are powers of 2, pad if needed */
export function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}
