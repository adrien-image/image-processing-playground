/**
 * The lossy core of JPEG, on grayscale images:
 *
 *   1. cut the image into 8×8 blocks and shift values from 0…255 to −128…127
 *   2. DCT: rewrite each block as a weighted sum of 64 cosine patterns (no loss yet)
 *   3. quantise: divide each weight by a step from the quantisation table and round —
 *      this is where information is thrown away. Fine detail (high frequencies) gets
 *      large steps, because the eye barely notices it; most of those weights become 0
 *   4. decode: multiply back, inverse DCT, shift back
 *
 * The real format then writes the quantised weights in zigzag order (low to high
 * frequency) with run-length and Huffman coding. `estimateBits` estimates that size from
 * the entropy of the same symbols, without building the actual file.
 */

export const BLOCK = 8;

/** The standard JPEG luminance quantisation table (quality 50), row-major. */
export const LUMA_QUANT = [
  16, 11, 10, 16, 24, 40, 51, 61,
  12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77,
  24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101,
  72, 92, 95, 98, 112, 100, 103, 99,
];

/** The quantisation table for a quality of 1–100 (the libjpeg / IJG scaling). */
export function quantTable(quality: number): number[] {
  const q = Math.min(100, Math.max(1, Math.round(quality)));
  const scale = q < 50 ? 5000 / q : 200 - 2 * q;
  return LUMA_QUANT.map(v => Math.min(255, Math.max(1, Math.floor((v * scale + 50) / 100))));
}

/** C[k][n] = α(k)·cos((2n + 1)kπ / 16): the orthonormal 8-point DCT-II matrix. */
const DCT = Array.from({ length: BLOCK }, (_, k) =>
  Array.from({ length: BLOCK }, (_, n) => (k === 0 ? Math.sqrt(1 / BLOCK) : Math.sqrt(2 / BLOCK)) * Math.cos(((2 * n + 1) * k * Math.PI) / (2 * BLOCK))),
);

/** 2-D DCT of one 8×8 block (row-major): C · B · Cᵀ. */
export function dct2(block: ArrayLike<number>): Float64Array {
  return separable(block, false);
}

/** Inverse 2-D DCT: Cᵀ · D · C. */
export function idct2(coefficients: ArrayLike<number>): Float64Array {
  return separable(coefficients, true);
}

function separable(input: ArrayLike<number>, inverse: boolean): Float64Array {
  const tmp = new Float64Array(64);
  const out = new Float64Array(64);
  const m = (a: number, b: number) => (inverse ? DCT[b][a] : DCT[a][b]);
  for (let r = 0; r < 8; r++) for (let k = 0; k < 8; k++) {
    let s = 0;
    for (let n = 0; n < 8; n++) s += m(k, n) * input[r * 8 + n];
    tmp[r * 8 + k] = s;   // rows transformed
  }
  for (let c = 0; c < 8; c++) for (let k = 0; k < 8; k++) {
    let s = 0;
    for (let n = 0; n < 8; n++) s += m(k, n) * tmp[n * 8 + c];
    out[k * 8 + c] = s;   // then columns
  }
  return out;
}

/** The 64 block positions in zigzag order: index into the 8×8 block for each position along the zigzag. */
export const ZIGZAG: number[] = (() => {
  const order: number[] = [];
  for (let s = 0; s < 15; s++) {
    const cells: number[] = [];
    for (let y = 0; y < 8; y++) {
      const x = s - y;
      if (x >= 0 && x < 8) cells.push(y * 8 + x);
    }
    // Odd diagonals run top-right → bottom-left, even ones the other way.
    order.push(...(s % 2 === 0 ? cells.reverse() : cells));
  }
  return order;
})();

export interface JpegResult {
  /** The decoded image, values 0–255. */
  output: Float64Array;
  /** Quantised coefficients, 64 per block, blocks in raster order (row-major inside each block). */
  quantised: Int16Array;
  blocksX: number;
  blocksY: number;
  /** How many of the quantised coefficients are not zero. */
  nonZero: number;
}

/** Compress and decompress a grayscale image (0–255). Edges are padded by repeating the last row/column. */
export function jpeg(gray: Float64Array, width: number, height: number, quality: number): JpegResult {
  const table = quantTable(quality);
  const blocksX = Math.ceil(width / BLOCK), blocksY = Math.ceil(height / BLOCK);
  const quantised = new Int16Array(blocksX * blocksY * 64);
  const output = new Float64Array(width * height);
  const block = new Float64Array(64);
  let nonZero = 0;
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        const sx = Math.min(width - 1, bx * 8 + x), sy = Math.min(height - 1, by * 8 + y);
        block[y * 8 + x] = gray[sy * width + sx] - 128;
      }
      const coeffs = dct2(block);
      const q = quantised.subarray((by * blocksX + bx) * 64, (by * blocksX + bx + 1) * 64);
      const dequantised = new Float64Array(64);
      for (let i = 0; i < 64; i++) {
        q[i] = Math.round(coeffs[i] / table[i]);
        if (q[i] !== 0) nonZero++;
        dequantised[i] = q[i] * table[i];
      }
      const pixels = idct2(dequantised);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        const px = bx * 8 + x, py = by * 8 + y;
        if (px < width && py < height) output[py * width + px] = Math.min(255, Math.max(0, Math.round(pixels[y * 8 + x] + 128)));
      }
    }
  }
  return { output, quantised, blocksX, blocksY, nonZero };
}

/** Number of bits needed to write |v| (JPEG's "size category"); 0 for 0. */
export const category = (v: number) => (v === 0 ? 0 : Math.floor(Math.log2(Math.abs(v))) + 1);

/**
 * Estimated size in bits of the entropy-coded data, using JPEG's own symbols:
 *   DC: the difference from the previous block's DC, coded as (size category) + size extra bits
 *   AC: in zigzag order, each non-zero value as (run of zeros before it, size category) +
 *       size extra bits; 16 zeros in a row as ZRL; the trailing zeros as one EOB symbol.
 * The symbols cost their Shannon entropy (what an optimal Huffman code approaches).
 */
export function estimateBits(result: JpegResult): number {
  const dcSymbols = new Map<number, number>();
  const acSymbols = new Map<number, number>();
  const count = (m: Map<number, number>, k: number) => m.set(k, (m.get(k) ?? 0) + 1);
  let extraBits = 0;
  let previousDc = 0;
  const blocks = result.blocksX * result.blocksY;
  for (let b = 0; b < blocks; b++) {
    const q = result.quantised.subarray(b * 64, (b + 1) * 64);
    const diff = q[0] - previousDc;
    previousDc = q[0];
    count(dcSymbols, category(diff));
    extraBits += category(diff);
    let run = 0;
    let last = 63;
    while (last > 0 && q[ZIGZAG[last]] === 0) last--;
    for (let k = 1; k <= last; k++) {
      const v = q[ZIGZAG[k]];
      if (v === 0) {
        run++;
        if (run === 16) { count(acSymbols, 0xf0); run = 0; }   // ZRL: sixteen zeros
        continue;
      }
      count(acSymbols, (run << 4) | category(v));
      extraBits += category(v);
      run = 0;
    }
    if (last < 63) count(acSymbols, 0x00);                       // EOB
  }
  const entropyBits = (m: Map<number, number>) => {
    let total = 0;
    m.forEach(c => { total += c; });
    let bits = 0;
    m.forEach(c => { bits += c * Math.max(1, -Math.log2(c / total)); });   // at least 1 bit per symbol
    return bits;
  };
  return entropyBits(dcSymbols) + entropyBits(acSymbols) + extraBits;
}
